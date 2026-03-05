import pretty_errors
import math
import decimal
import itertools
from decimal import Decimal
from fractions import Fraction
import bisect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter


# ══════════════════════════════════════════════════════════════════════════════
# ── 量子化ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

# ── Linear Quantization ───────────────────────────────────────────────────────
def linear_quantize(input, scale, zero_point, is_weight):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    Parameters:
    ----------
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if is_weight:
        if len(input.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        elif len(input.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
    else:
        if len(input.shape) == 2:
            scale = scale.view(1, -1)
            zero_point = zero_point.view(1, -1)
        elif len(input.shape) == 3:
            scale = scale.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        elif len(input.shape) == 4:
            scale = scale.view(1, -1, 1, 1)
            zero_point = zero_point.view(1, -1, 1, 1)
        else:
            raise NotImplementedError

    # quantized = float / scale + zero_point
    return torch.round(1. / scale * input + zero_point)


# ── Symmetric Quantization Scale ──────────────────────────────────────────────
def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps

        max_val = torch.max(-min_val, max_val)
        scale = max_val / float(n)
        scale.clamp_(eps)

    return scale


# ── Symmetric Quantization Function ───────────────────────────────────────────
class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        scale = specified_scale
        zero_point = torch.tensor(0., device=x.device)
        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, is_weight=is_weight)
        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)
        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        is_weight = ctx.is_weight
        if is_weight:
            if len(grad_output.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
            elif len(grad_output.shape) == 2:
                scale = scale.view(-1, 1)
            else:
                scale = scale.view(-1)
        else:
            if len(grad_output.shape) == 2:
                scale = scale.view(1, -1)
            elif len(grad_output.shape) == 3:
                scale = scale.view(1, 1, -1)
            elif len(grad_output.shape) == 4:
                scale = scale.view(1, -1, 1, 1)
            else:
                raise NotImplementedError
        return grad_output.clone() / scale, None, None, None


# ── Ternary Quantization Function ─────────────────────────────────────────────
class TernaryQuantFunction(Function):
    """
    SymmetricQuantFunction と同一構造だが、clamp を固定で [-1, 1] にすることで
    {-1, 0, +1} の３値量子化を行う。STE の backward も SymmetricQuantFunction と同じ。
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        scale = specified_scale
        zero_point = torch.tensor(0., device=x.device)
        new_quant_x = linear_quantize(x, scale, zero_point, is_weight=is_weight)
        new_quant_x = torch.clamp(new_quant_x, -1, 1)  # {-1, 0, +1} 固定
        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        is_weight = ctx.is_weight
        if is_weight:
            if len(grad_output.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
            elif len(grad_output.shape) == 2:
                scale = scale.view(-1, 1)
            else:
                scale = scale.view(-1)
        else:
            if len(grad_output.shape) == 2:
                scale = scale.view(1, -1)
            elif len(grad_output.shape) == 3:
                scale = scale.view(1, 1, -1)
            elif len(grad_output.shape) == 4:
                scale = scale.view(1, -1, 1, 1)
            else:
                raise NotImplementedError
        return grad_output.clone() / scale, None, None, None


# ── Floor Function (STE) ──────────────────────────────────────────────────────
class floor_ste(Function):
    """Straight-through Estimator(STE) for torch.floor()"""

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


# ── Round Function (STE) ──────────────────────────────────────────────────────
class round_ste(Function):
    """Straight-through Estimator(STE) for torch.round()"""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


# ── Batch Frexp ───────────────────────────────────────────────────────────────
def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.
    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """
    shape_of_input = inputs.size()
    device = inputs.device
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** max_bit)).quantize(Decimal('1'),
                                                                  rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    output_e = float(max_bit) - output_e

    return torch.from_numpy(output_m).to(device).view(shape_of_input), \
           torch.from_numpy(output_e).to(device).view(shape_of_input)


# ── Fixed-point Multiply ──────────────────────────────────────────────────────
class fixedpoint_mul(Function):
    """
    Function to perform fixed-point arthmetic that can match integer arthmetic on hardware.
    Parameters:
    ----------
    pre_act: input tensor
    pre_act_scaling_factor: the scaling factor of the input tensor
    bit_num: quantization bitwidth
    quant_mode: The mode for quantization, 'symmetric' or 'asymmetric'
    z_scaling_factor: the scaling factor of the output tensor
    identity: identity tensor
    identity_scaling_factor: the scaling factor of the identity tensor
    """

    @staticmethod
    def forward(ctx, pre_act, pre_act_scaling_factor,
                bit_num, quant_mode, z_scaling_factor,
                identity=None, identity_scaling_factor=None):

        if len(pre_act.shape) == 2:
            reshape = lambda x: x.view(1, -1)
        elif len(pre_act.shape) == 3:
            reshape = lambda x: x.view(1, 1, -1)
        elif len(pre_act.shape) == 4:
            reshape = lambda x: x.view(1, -1, 1, 1)
        else:
            raise NotImplementedError
        ctx.identity = identity

        if quant_mode == 'symmetric':
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2 ** bit_num - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor

            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0 ** e))

            if identity is not None:
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0 ** e1))

                output = output1 + output

            if bit_num in [4, 8, 16, 32]:
                if quant_mode == 'symmetric':
                    return torch.clamp(output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp(output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, \
               identity_grad, None


# ══════════════════════════════════════════════════════════════════════════════
# ── 量子化レイヤー
# ══════════════════════════════════════════════════════════════════════════════

# ── Quantized Linear Layer ────────────────────────────────────────────────────
class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric'):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True)
        else:
            self.bias_integer = None

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
               * bias_scaling_factor, bias_scaling_factor


# ── Ternary Linear Layer ──────────────────────────────────────────────────────
class TerLinear(nn.Linear):
    """
    ３値重み量子化線形層。

    重みをチャネルごとの絶対値平均でスケーリングし、round→clip により {-1, 0, +1} に量子化する。
    活性化は QuantLinear と同様に 8-bit 対称量子化（prev_act_scaling_factor を外部から受け取る）。
    use_layernorm=True のとき、forward の先頭で IntLayerNorm を実行する。

    Parameters:
    ----------
    in_features : int
    out_features : int
    bias : bool, default True
    weight_bit : int, default 2
        名目上のビット幅（実際は {-1,0,+1} の 3 値固定）。
    bias_bit : int, default 32
        バイアスの量子化ビット幅。
    per_channel : bool, default True
        重みのスケーリングをチャネルごとに行うか（True のみサポート）。
    quant_mode : str, default 'symmetric'
        量子化モード。
    use_layernorm : bool, default False
        True のとき forward の先頭で IntLayerNorm を実行する。
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=2,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric',
                 use_layernorm=False):
        super(TerLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.use_layernorm = use_layernorm

        if self.quant_mode == "symmetric":
            self.weight_function = TernaryQuantFunction.apply
            self.bias_function   = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        if self.use_layernorm:
            self.layernorm = IntLayerNorm(in_features)
            self.qact_norm = QuantAct(activation_bit=8, quant_mode=quant_mode)

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(TerLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={}, use_layernorm={})".format(
            self.weight_bit, self.quant_mode, self.use_layernorm)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        if self.use_layernorm:
            x, prev_act_scaling_factor = self.layernorm(x, prev_act_scaling_factor)
            x, prev_act_scaling_factor = self.qact_norm(x, prev_act_scaling_factor)

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                abs_mean = v.abs().mean(axis=1)
                self.fc_scaling_factor = abs_mean.clamp(min=torch.finfo(torch.float32).eps)
            else:
                raise Exception('For weight, we only support per_channel quantization.')

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.bias_function(
                self.bias, self.bias_bit, bias_scaling_factor, True)
        else:
            self.bias_integer = None

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
               * bias_scaling_factor, bias_scaling_factor


# ── Quantized Activation Layer ────────────────────────────────────────────────
class QuantAct(nn.Module):
    """
    Class to quantize given activations
    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'symmetric'
        The mode for quantization.
    """

    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 quant_mode="symmetric"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel

        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, quant_mode: {2}, Act_min: {3:.2f}, Act_max: {4:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.quant_mode, self.min_val.item(), self.max_val.item())

    def fix(self):
        """fix the activation range by setting running stat"""
        self.running_stat = False

    def unfix(self):
        """unfix the activation range by setting running stat"""
        self.running_stat = True

    def forward(self, x,
                pre_act_scaling_factor=None,
                identity=None,
                identity_scaling_factor=None):
        with torch.no_grad():
            x_act = x if identity is None else identity + x
            if self.running_stat:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = self.min_val * self.act_range_momentum + \
                                   cur_min * (1 - self.act_range_momentum)
                    self.max_val = self.max_val * self.act_range_momentum + \
                                   cur_max * (1 - self.act_range_momentum)
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val)

        if pre_act_scaling_factor is None:
            quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor, False)
        else:
            quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor,
                self.activation_bit, self.quant_mode,
                self.act_scaling_factor,
                identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)
        return quant_act_int * correct_output_scale, self.act_scaling_factor


# ── Quantized MatMul Layer ────────────────────────────────────────────────────
class QuantMatMul(nn.Module):
    """Class to quantize weights of given matmul layer"""

    def __init__(self):
        super(QuantMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


# ── Quantized Conv2d Layer ────────────────────────────────────────────────────
class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 8
        Bitwidth for quantized weights.
    bias_bit : int, default 32
        Bitwidth for quantized bias.
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default True
        Whether to use channel-wise quantization.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 quant_mode="symmetric",
                 per_channel=True):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)

        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, pre_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True)
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True)
        else:
            self.bias_integer = None

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.stride, self.padding,
                         self.dilation, self.groups) * correct_output_scale, correct_output_scale)


# ── Quantized BN+Conv2d Layer ─────────────────────────────────────────────────
class QuantBNConv2d(nn.Module):
    """
    BN を畳み込みに折り畳んだ量子化 Conv2d 層。

    Parameters
    ----------
    in_channels, out_channels, kernel_size : 通常の Conv2d と同じ。
    weight_bit : int
        重みの量子化ビット幅。
    bias_bit : int
        バイアスの量子化ビット幅。
    quant_mode : str
        'symmetric' のみ対応。
    per_channel : bool
        チャネルごと量子化するか（現在 per-tensor 固定）。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 weight_bit: int = 8,
                 bias_bit: int = 32,
                 quant_mode: str = "symmetric",
                 per_channel: bool = True):
        super().__init__()
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ks
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *ks))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel

        self.register_buffer('conv_scaling_factor', torch.zeros(out_channels))

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
            self.bias_function = SymmetricQuantFunction.apply
        else:
            raise ValueError(f"unknown quant mode: {quant_mode}")

    def __repr__(self):
        return (
            f"QuantBNConv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, "
            f"weight_bit={self.weight_bit})"
        )

    def _fold_bn(self, mu, sigma2):
        W = self.weight
        gamma = self.bn.weight
        beta = self.bn.bias
        sigma = (sigma2 + self.bn.eps).sqrt()
        s = gamma / sigma
        return W * s.view(-1, 1, 1, 1), beta - mu * s

    def forward(self, x: torch.Tensor, pre_act_scaling_factor: torch.Tensor):
        # 入力のスケール
        pre_act_sf = pre_act_scaling_factor.view(1)
        
        #? 学習時はConv2d -> 平均分散を取得 -> BNで内部runnning_mead, varを更新 
        if self.training:
            y_fp = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
            with torch.no_grad():
                _ = self.bn(y_fp.detach())
            mu = y_fp.mean(dim=[0, 2, 3])
            sigma2 = y_fp.var(dim=[0, 2, 3], unbiased=False)
        else:
            #! eval時は学習しているrunning_mean, varを使う
            mu = self.bn.running_mean
            sigma2 = self.bn.running_var

        # BN fold
        W_fold, b_fold = self._fold_bn(mu, sigma2)

        # 重みの量子化スケール取得
        with torch.no_grad():
            if self.per_channel:
                v = W_fold.detach().reshape(W_fold.shape[0], -1)
                min_val = v.min(axis=1).values
                max_val = v.max(axis=1).values
                self.conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, min_val, max_val)
            else:
                v = W_fold.detach().reshape(-1)
                self.conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, v.min().unsqueeze(0), v.max().unsqueeze(0))
        # 重みを量子化
        weight_integer = self.weight_function(W_fold, self.weight_bit, self.conv_scaling_factor, True)
        # biasの量子化スケール取得
        bias_scaling_factor = self.conv_scaling_factor * pre_act_sf
        # biasの量子化
        bias_integer = self.bias_function(b_fold, self.bias_bit, bias_scaling_factor, True)
        # 入力xを整数に戻す
        x_int = round_ste.apply(x / pre_act_sf.view(1, 1, 1, 1))
        # スケール
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)
        # 畳み込み
        out_int = F.conv2d(x_int, weight_integer, bias_integer, self.stride, self.padding, self.dilation, self.groups)
        # return
        return out_int * correct_output_scale, correct_output_scale


# ── Integer Layer Norm ────────────────────────────────────────────────────────
class IntLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-5,
                 elementwise_affine=True):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer('norm_scaling_factor', torch.zeros(1))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(x.device)

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2 ** 31 - 1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor


# ── Integer GELU ──────────────────────────────────────────────────────────────
class IntGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit
        self.n = 23  # sufficiently large integer
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n

        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste.apply((2 ** 31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        sigmoid_scaling_factor = torch.tensor([1 / 2 ** (self.output_bit - 1)], device=x.device)

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


# ── Integer Softmax ───────────────────────────────────────────────────────────
class IntSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.n = 15  # sufficiently large integer
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste.apply((2 ** 31 - 1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        scaling_factor = torch.tensor([1 / 2 ** (self.output_bit - 1)], device=x.device)

        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor


# ══════════════════════════════════════════════════════════════════════════════
# ── 量子化 TinyEFormer module
# ══════════════════════════════════════════════════════════════════════════════

# ── DropPath ──────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """
    Stochastic Depth（確率的深さ）。
    学習中のみ、ブロック全体を確率 drop_prob でゼロにする。
    （tiny_eformer_model.py より転記）
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        # drop_prob=0 なら何もしない Identity 相当になる
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 評価時 or drop_prob=0 はそのまま通す
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # バッチ次元だけ残してすべて 1 のマスク形状を作る
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Bernoulli サンプル（一様乱数を keep_prob で丸める）
        r = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * r.floor_()


# ── QStem ─────────────────────────────────────────────────────────────────────
class QStem(nn.Module):
    """
    量子化 Stem（tiny_eformer Stem の QAT 版）。
    Conv3x3(s=2) + BN + ReLU → Conv3x3(s=2) + BN + ReLU。
    入力 (B, in_chs, H, W) → 出力 (B, out_chs, H/4, W/4)。
    各演算は (テンソル, スケーリングファクタ) のタプルで伝播する。
    """
    def __init__(self, in_chs: int, out_chs: int,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        # 中間チャネル数（out_chs の半分）
        mid = out_chs // 2
        # 入力 float を int8 に量子化する（EMA でスケール追跡）
        self.qact_in  = QuantAct(activation_bit=act_bit)
        # Conv1: in_chs → mid, stride=2, BN を重みに折り畳み済み
        self.conv1 = QuantBNConv2d(in_chs, mid, kernel_size=3, stride=2, padding=1, weight_bit=weight_bit)
        # Conv1 後の活性化
        self.relu1 = nn.ReLU6(inplace=True)
        # Conv1 出力（int32 相当）を int8 へ再量子化
        self.qact1    = QuantAct(activation_bit=act_bit)
        # Conv2: mid → out_chs, stride=2, BN を重みに折り畳み済み
        self.conv2 = QuantBNConv2d(mid, out_chs, kernel_size=3, stride=2, padding=1, weight_bit=weight_bit)
        # Conv2 後の活性化
        self.relu2 = nn.ReLU6(inplace=True)
        # Conv2 出力（int32 相当）を int8 へ再量子化
        self.qact2    = QuantAct(activation_bit=16)

    def forward(self, x: torch.Tensor, sf=None):
        # 入力 float → int8 へ量子化（pre_act_scaling_factor=None: EMA で直接スケール推定）
        x, sf = self.qact_in(x)
        # Conv1 + BN fold: int8 × int8-weight → int32 相当
        x, sf = self.conv1(x, sf)
        # ReLU6: 負値をゼロにクランプし上限を 6 に制限（スケールは変化しない）
        x = self.relu1(x)
        # int32 → int8 へ再量子化（fixedpoint_mul でスケールを揃える）
        x, sf = self.qact1(x, sf)
        # Conv2 + BN fold: int8 × int8-weight → int32 相当
        x, sf = self.conv2(x, sf)
        # ReLU6
        x = self.relu2(x)
        # int32 → int8 へ再量子化
        x, sf = self.qact2(x, sf)
        return x, sf


# ── QMlp ─────────────────────────────────────────────────────────────────────
class QMlp(nn.Module):
    """
    量子化 MLP（tiny_eformer Mlp の QAT 版、mid_conv=True 相当）。
    Conv1×1(BN) → ReLU → DW-Conv3×3(BN) → ReLU → Conv1×1(BN)。
    in_features → hidden_features → out_features の 2 段変換。
    """
    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int = None, drop: float = 0.,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        out_features = out_features or in_features
        # fc1: 1×1 conv で通路を拡張（in → hidden）
        self.fc1 = QuantBNConv2d(in_features, hidden_features, kernel_size=1, weight_bit=weight_bit)
        # fc1 後の活性化
        self.relu1 = nn.ReLU6(inplace=True)
        # fc1 出力（int32）を int8 へ再量子化
        self.qact1  = QuantAct(activation_bit=act_bit)
        # mid: 3×3 depth-wise conv で空間特徴を混合（hidden → hidden, groups=hidden）
        self.mid = QuantBNConv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, weight_bit=weight_bit)
        # mid 後の活性化
        self.relu2 = nn.ReLU6(inplace=True)
        # mid 出力（int32）を int8 へ再量子化
        self.qact2  = QuantAct(activation_bit=act_bit)
        # fc2: 1×1 conv で通路を圧縮（hidden → out）
        self.fc2 = QuantBNConv2d(hidden_features, out_features, kernel_size=1, weight_bit=weight_bit)
        # Dropout（float 演算; スケールに影響しない）
        self.drop   = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        # fc1 + BN fold: int8 × int8-weight → int32 相当（チャネル拡張）
        x, sf = self.fc1(x, sf)
        # ReLU6（GELU より整数演算に馴染む）
        x = self.relu1(x)
        self.drop(x)
        # int32 → int8 へ再量子化
        x, sf = self.qact1(x, sf)
        # DW-Conv + BN fold: int8 × int8-weight → int32 相当（空間特徴の局所混合）
        x, sf = self.mid(x, sf)
        x = self.relu2(x)
        # int32 → int8 へ再量子化
        x, sf = self.qact2(x, sf)
        # fc2 + BN fold: int8 × int8-weight → int32 相当（チャネル圧縮）
        x, sf = self.fc2(x, sf)
        # fc2 後は呼び出し元（QFFN 等）が QuantAct で仕上げる
        return x, sf


# ── QFFN ─────────────────────────────────────────────────────────────────────
class QFFN(nn.Module):
    """
    量子化 FFN ブロック（tiny_eformer FFN の QAT 版）。
    x + drop_path(layer_scale × QMlp(x)) の残差構造。
    残差加算は QuantAct の identity 引数を使い、2 つのスケールを揃えてから加算する。
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init: float = 1e-5,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        
        # 量子化
        self.qact      = QuantAct(8) 
        # MLP 本体（fc1 → DW-Conv → fc2）
        self.mlp       = QMlp(dim, int(dim * mlp_ratio), dim, drop=drop, weight_bit=weight_bit, act_bit=act_bit)
        # Stochastic Depth（学習時のみランダムにゼロ化）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # layer scale パラメータ（V2 の特徴; 初期値が小さく残差を支配させる）
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init * torch.ones(dim, 1, 1), requires_grad=True)
        # 残差加算後を int8 へ再量子化（identity 引数で skip connection を融合）
        self.qact_out  = QuantAct(activation_bit=act_bit)

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        # mlp用量子化
        x_0, sf_0 = x, sf
        x, sf = self.qact(x, sf)
        # MLP フォワード: (x_int8, sf) → (mlp_out_int32, sf_mlp)
        mlp_out, sf_mlp = self.mlp(x, sf)
        # layer scale でスケーリング（float の scalar 積; 量子化スケールには影響しない近似）
        if self.use_layer_scale:
            mlp_out = mlp_out * self.layer_scale
        # Stochastic Depth（学習時のみ確率的にゼロにする）
        mlp_out = self.drop_path(mlp_out)
        # 残差加算: x + mlp_out を QuantAct で int8 に再量子化
        # identity=x, identity_scaling_factor=sf を渡すと fixedpoint_mul が
        # mlp_out と x のスケールを共通の出力スケールへ揃えてから加算する
        x, sf = self.qact_out(mlp_out, sf_mlp, identity=x_0, identity_scaling_factor=sf_0)
        return x, sf


# ── QAttention4D ─────────────────────────────────────────────────────────────
class QAttention4D(nn.Module):
    """
    量子化 Attention4D（tiny_eformer Attention4D の QAT 版）。
    Q / K / V は QuantBNConv2d で生成し、QuantMatMul でアテンション計算を行う。
    attention biases は float パラメータのまま加算（位置エンコードの代替）。
    stride 指定時は stride_conv でダウンサンプルし、出力を bilinear でアップサンプルする。
    """
    def __init__(self, dim: int = 384, key_dim: int = 16, num_heads: int = 4,
                 attn_ratio: int = 4, resolution: int = 7, stride: int = None,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim   = key_dim
        # Q @ K^T のスケーリング係数（key_dim の -0.5 乗）
        self.scale     = key_dim ** -0.5
        self.d         = int(attn_ratio * key_dim)
        # V の全ヘッド合計次元
        self.dh        = self.d * num_heads
        # Q / K の全ヘッド合計次元
        self.nh_kd     = key_dim * num_heads

        # stride がある場合: stride_conv でダウンサンプル後 upsample で戻す
        if stride is not None:
            # アテンション計算時の空間解像度（切り上げ）
            self.resolution  = math.ceil(resolution / stride)
            # QuantBNConv2d in8 * int8 -> int32 -> int32
            self.stride_conv = QuantBNConv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)
            # stride_conv の出力を int8 へ量子化
            self.qact_stride = QuantAct(activation_bit=act_bit)
            # 最後に元解像度へ戻す bilinear upsample
            self.upsample    = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        else:
            self.resolution  = resolution
            self.stride_conv = None
            self.upsample    = None

        # アテンション計算における系列長（resolution × resolution）
        self.N = self.resolution ** 2

        # 1. Q 生成: Conv1×1 + BN fold（dim → nh_kd）[int8 → int32]
        self.q       = QuantBNConv2d(dim, self.nh_kd, kernel_size=1, weight_bit=weight_bit)
        # Q を int8 へ再量子化
        self.qact_q  = QuantAct(activation_bit=act_bit)

        # 2. K 生成: Conv1×1 + BN fold（dim → nh_kd）[int8 → int32]
        self.k       = QuantBNConv2d(dim, self.nh_kd, kernel_size=1, weight_bit=weight_bit)
        # K を int8 へ再量子化
        self.qact_k  = QuantAct(activation_bit=act_bit)

        # 3. V 生成: Conv1×1 + BN fold（dim → dh）[int8 → int32]
        self.v       = QuantBNConv2d(dim, self.dh, kernel_size=1, weight_bit=weight_bit)
        # V を int8 へ再量子化
        self.qact_v  = QuantAct(activation_bit=act_bit)

        # 4. v_local: V に DW-Conv3×3 を適用して空間局所構造を補完 [int8 → int32]
        self.v_local = QuantBNConv2d(self.dh, self.dh, kernel_size=3, stride=1, padding=1, groups=self.dh, weight_bit=weight_bit)
        # v_local を int8 へ再量子化
        self.qact_vlocal = QuantAct(activation_bit=act_bit)

        #5.0 量子化
        self.qact_head1    = QuantAct(8)
        # 5. talking head 1: ヘッド間 Conv1×1 でスコアを混合（float で実行）
        self.talking_head1 = QuantConv2d(num_heads, num_heads, 1, bias=False)
        # 6.0 量子化
        self.qact_head2    = QuantAct(8)
        # 6. talking head 2: softmax 後のヘッド間混合（float で実行）
        self.talking_head2 = QuantConv2d(num_heads, num_heads, 1, bias=False)

        # Q @ K^T の行列積モジュール [int8 × int8 → int32]
        self.qmatmul_qk      = QuantMatMul()
        # attention score を softmax 直前に int8 へ再量子化
        self.qact_pre_softmax = QuantAct(activation_bit=act_bit)
        # 整数近似 softmax
        self.int_softmax     = IntSoftmax(output_bit=act_bit)
        # softmax 後スコアを int8 へ再量子化
        self.qact_post_softmax = QuantAct(activation_bit=act_bit)
        # attn × V の行列積モジュール [int8 × int8 → int32]
        self.qmatmul_attnv   = QuantMatMul()
        # v_local と attn@V の加算後を int8 へ再量子化（identity で2スケールを融合）
        self.qact_merge      = QuantAct(activation_bit=act_bit)

        # 7. 出力 projection: ReLU → Conv1×1 + BN fold（dh → dim）[int8 → int32]
        self.proj_act  = nn.ReLU6(inplace=True)
        self.proj_conv = QuantBNConv2d(self.dh, dim, kernel_size=1, weight_bit=weight_bit)
        # projection 出力を int8 へ再量子化
        self.qact_out  = QuantAct(activation_bit=16)

        # attention bias（学習可能パラメータ; 位置エンコードの代替; float のまま加算）
        pts = list(itertools.product(range(self.resolution), range(self.resolution)))
        attention_offsets: dict = {}
        idxs: list = []
        for p1 in pts:
            for p2 in pts:
                off = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if off not in attention_offsets:
                    attention_offsets[off] = len(attention_offsets)
                idxs.append(attention_offsets[off])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(self.N, self.N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        # 学習モード: キャッシュを削除して毎回 biases テーブルを引く
        if mode and hasattr(self, 'ab'):
            del self.ab
        # 評価モード: attention_biases をあらかじめ展開してキャッシュ
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        B, C, H, W = x.shape

        # stride_conv がある場合は空間解像度を落としてから QKV を生成する
        if self.stride_conv is not None:
            # stride_conv 
            xs = self.stride_conv(x)
            # stride_conv 出力を int8 へ量子化
            xs, sf_s = self.qact_stride(xs)
        else:
            # stride なし: そのまま使う
            xs, sf_s = x, sf

        # ── Q 生成 ────────────────────────────────────────────────────────────
        # Conv1×1 + BN fold: int8 × int8-weight → int32 相当
        q, sf_q = self.q(xs, sf_s)
        # int32 → int8 へ再量子化
        q, sf_q = self.qact_q(q, sf_q)

        # ── K 生成 ────────────────────────────────────────────────────────────
        # Conv1×1 + BN fold: int8 × int8-weight → int32 相当
        k, sf_k = self.k(xs, sf_s)
        # int32 → int8 へ再量子化
        k, sf_k = self.qact_k(k, sf_k)

        # ── V 生成 ────────────────────────────────────────────────────────────
        # Conv1×1 + BN fold: int8 × int8-weight → int32 相当
        v, sf_v = self.v(xs, sf_s)
        # int32 → int8 へ再量子化
        v, sf_v = self.qact_v(v, sf_v)

        # ── v_local 生成 ──────────────────────────────────────────────────────
        # DW-Conv3×3 + BN fold: int8 × int8-weight → int32 相当（V の空間局所成分）
        v_local, sf_vl = self.v_local(v, sf_v)
        # int32 → int8 へ再量子化
        v_local, sf_vl = self.qact_vlocal(v_local, sf_vl)

        # ── Q / K / V を系列 (B, heads, N, dim) 形状に変換 ───────────────────
        # Q: (B, nh_kd, res, res) → (B, heads, N, key_dim)
        q     = q.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        # K: (B, nh_kd, res, res) → (B, heads, key_dim, N) ← 転置済み
        k     = k.flatten(2).reshape(B, self.num_heads, -1, self.N)
        # V: (B, dh, res, res) → (B, heads, N, d)
        v_seq = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        # v_local を v と同じ空間形状のまま保持（後で加算）
        v_local_4d = v_local  # (B, dh, res, res)

        # ── アテンションスコア計算 ─────────────────────────────────────────────
        # Q @ K^T （QuantMatMul）: int8 × int8 → int32 相当 [B, heads, N, N]
        attn, sf_attn = self.qmatmul_qk(q, sf_q, k, sf_k)
        # key_dim^-0.5 でスケーリング（float 乗算; 定数スケール変換）
        attn    = attn * self.scale
        sf_attn = sf_attn * self.scale 
        # attention biases を float のまま加算（位置エンコード代替; 学習可能パラメータ）
        ab = (self.ab if hasattr(self, 'ab') else self.attention_biases[:, self.attention_bias_idxs])
        attn = attn + ab.to(attn.device)
        # 量子化
        attn, sf_attn = self.qact_head1(attn, sf_attn)
        # talking head 1: ヘッド間 Conv1×1 でスコアを混合
        attn, sf_attn = self.talking_head1(attn, sf_attn)
        # softmax 直前に int8 へ再量子化
        attn, sf_attn = self.qact_pre_softmax(attn, sf_attn)
        # IntSoftmax: 整数近似 softmax [int8 → int8]
        attn, sf_attn = self.int_softmax(attn, sf_attn)
        # softmax 後を int8 へ再量子化
        attn, sf_attn = self.qact_post_softmax(attn, sf_attn)
        # talking head 2: softmax 後のヘッド間混合（float Conv2d）
        attn, sf_attn = self.talking_head2(attn, sf_attn)
        # talking head 2 後は float 操作が入るので再び int8 へ量子化し直す
        attn, sf_attn = self.qact_post_softmax(attn, sf_attn)

        # ── attn @ V ─────────────────────────────────────────────────────────
        # （QuantMatMul）: int8 × int8 → int32 相当 [B, heads, N, d]
        out, sf_out = self.qmatmul_attnv(attn, sf_attn, v_seq, sf_v)

        # (B, heads, N, d) → (B, dh, res, res) に形状を戻す
        out = out.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution)

        # v_local との加算: QuantAct の identity 引数で2スケールを揃えて融合 [int8]
        out, sf_out = self.qact_merge(out, sf_out, identity=v_local_4d, identity_scaling_factor=sf_vl)

        # stride がある場合 bilinear upsample で元解像度へ戻す（值はそのまま）
        if self.upsample is not None:
            out = self.upsample(out)

        # ── 出力 projection ──────────────────────────────────────────────────
        # ReLU（非線形; スケールに影響しない）
        out = self.proj_act(out)
        # Conv1×1 + BN fold: int8 × int8-weight → int32 相当（dh → dim）
        out, sf_out = self.proj_conv(out, sf_out)
        # int32 → int8 へ再量子化
        out, sf_out = self.qact_out(out, sf_out)

        return out, sf_out


# ── QAttnFFN ─────────────────────────────────────────────────────────────────
class QAttnFFN(nn.Module):
    """
    量子化 AttnFFN ブロック（tiny_eformer AttnFFN の QAT 版）。
    token_mixer（QAttention4D）と QMlp を残差構造で繋ぐ。
    2 つの残差加算はそれぞれ QuantAct の identity 引数で処理する。
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init: float = 1e-5,
                 resolution: int = 7, stride: int = None,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        # 量子化
        self.qact_0 = QuantAct()
        self.qact_1 = QuantAct()
        # attention token mixer（QAttention4D; stride 指定でダウンサンプル対応）
        self.token_mixer = QAttention4D(dim, resolution=resolution, stride=stride, weight_bit=weight_bit, act_bit=act_bit)
        # MLP（fc1 → DW-Conv → fc2）
        self.mlp = QMlp(dim, int(dim * mlp_ratio), dim, drop=drop, weight_bit=weight_bit, act_bit=act_bit)
        # Stochastic Depth
        self.drop_path    = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # layer scale パラメータ（attention 残差用と MLP 残差用）
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init * torch.ones(dim, 1, 1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init * torch.ones(dim, 1, 1), requires_grad=True)
        # attention 残差後を int8 へ再量子化（skip + attn_out を融合）
        self.qact_attn_res = QuantAct(activation_bit=16)
        # MLP 残差後を int8 へ再量子化（skip + mlp_out を融合）
        self.qact_mlp_res  = QuantAct(activation_bit=act_bit)

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        # ── attention 残差 ───────────────────────────────────────────────────
        # attention入力
        x_0, sf_0 = x, sf
        x, sf = self.qact_0(x, sf)
        # token_mixer フォワード: (x_int8, sf) → (attn_out_int8, sf_attn)
        attn_out, sf_attn = self.token_mixer(x, sf)
        # layer scale でスケーリング（float 積; 近似的に量子化スケールに影響しない）
        if self.use_layer_scale:
            attn_out = attn_out * self.layer_scale_1
        # Stochastic Depth（学習時のみ確率的にゼロ化）
        attn_out = self.drop_path(attn_out)
        # x + attn_out を QuantAct で int8 へ再量子化（identity で2スケールを揃えて加算）
        x, sf = self.qact_attn_res(attn_out, sf_attn, identity=x_0, identity_scaling_factor=sf_0)

        # ── MLP 残差 ─────────────────────────────────────────────────────────
        # mlp用量子化
        x_0, sf_0 = x, sf
        x, sf = self.qact_1(x, sf)
        # MLP フォワード: (x_int8, sf) → (mlp_out_int32, sf_mlp)
        mlp_out, sf_mlp = self.mlp(x, sf)
        # layer scale でスケーリング
        if self.use_layer_scale:
            mlp_out = mlp_out * self.layer_scale_2
        # Stochastic Depth
        mlp_out = self.drop_path(mlp_out)
        # x + mlp_out を QuantAct で int8 へ再量子化
        x, sf = self.qact_mlp_res(mlp_out, sf_mlp, identity=x_0, identity_scaling_factor=sf_0)

        return x, sf


# ── QEmbedding ────────────────────────────────────────────────────────────────
class QEmbedding(nn.Module):
    """
    量子化 Embedding（ステージ間ダウンサンプリング; tiny_eformer Embedding の asub=False 版）。
    Conv(patch_size, stride=2) + BN fold → QuantAct。
    """
    def __init__(self, in_chs: int, out_chs: int,
                 patch_size: int = 3, stride: int = 2, padding: int = 1,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        # Conv(patch_size×patch_size, s=stride) + BN fold でダウンサンプル
        self.proj = QuantBNConv2d(in_chs, out_chs, kernel_size=patch_size, stride=stride, padding=padding, weight_bit=weight_bit)
        # proj 出力（int32）を int8 へ再量子化
        self.qact = QuantAct(activation_bit=act_bit)

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        # Conv + BN fold: int8 × int8-weight → int32 相当（空間縮小 + チャネル変換）
        x, sf = self.proj(x, sf)
        # int32 → int8 へ再量子化
        x, sf = self.qact(x, sf)
        return x, sf


# ── QLGQuery ──────────────────────────────────────────────────────────────────
class QLGQuery(nn.Module):
    """
    量子化 LGQuery（QAttention4DDownsample の Q ストリーム専用）。
    DW-Conv(s=2) と AvgPool(s=2) を加算し、Conv1×1+BN で channel 変換。
    （tiny_eformer LGQuery の QAT 版）
    """
    def __init__(self, in_dim: int, out_dim: int,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        # 空間局所特徴: DW-Conv(3×3, s=2)（float; BN は後段に含める）
        self.local = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim)
        self.local_bn = nn.BatchNorm2d(in_dim)
        # 平均プーリング（stride=2 でダウンサンプル; float 演算）
        self.pool    = nn.AvgPool2d(1, 2, 0)
        # 加算後の量子化
        self.qact_sum = QuantAct(activation_bit=act_bit)
        # channel 変換: Conv1×1 + BN fold（in_dim → out_dim）[int8 → int32]
        self.proj    = QuantBNConv2d(in_dim, out_dim, kernel_size=1, weight_bit=weight_bit)
        # proj 出力を int8 へ再量子化
        self.qact_out = QuantAct(activation_bit=act_bit)

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        # DW-Conv(s=2): float フォワード
        local_out = self.local_bn(self.local(x))
        # AvgPool(s=2): float フォワード
        pool_out  = self.pool(x)
        # 加算後を int8 へ量子化（EMA でスケール推定）
        fused, sf_fused = self.qact_sum(local_out + pool_out)
        # Conv1×1 + BN fold: int8 × int8-weight → int32 相当（channel 変換）
        out, sf_out = self.proj(fused, sf_fused)
        # int32 → int8 へ再量子化
        out, sf_out = self.qact_out(out, sf_out)
        return out, sf_out


# ── QAttention4DDownsample ────────────────────────────────────────────────────
class QAttention4DDownsample(nn.Module):
    """
    量子化 Attention4DDownsample（QEmbeddingAttn の attention ストリーム専用）。
    Q は QLGQuery（stride=2 ダウンサンプル）、K/V は元解像度 x から生成する。
    （tiny_eformer Attention4DDownsample の QAT 版）
    """
    def __init__(self, dim: int = 384, key_dim: int = 16, num_heads: int = 4,
                 attn_ratio: int = 4, resolution: int = 7, out_dim: int = None,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        self.num_heads   = num_heads
        self.scale       = key_dim ** -0.5
        self.key_dim     = key_dim
        self.d           = int(attn_ratio * key_dim)
        self.dh          = self.d * num_heads
        self.nh_kd       = key_dim * num_heads
        self.out_dim     = out_dim if out_dim is not None else dim
        self.resolution  = resolution
        # Q の空間解像度（stride=2 で半分）
        self.resolution2 = math.ceil(resolution / 2)
        self.N           = resolution ** 2
        self.N2          = self.resolution2 ** 2

        # Q: QLGQuery で stride=2 ダウンサンプル [int8 → int8]
        self.q = QLGQuery(dim, self.nh_kd, weight_bit=weight_bit, act_bit=act_bit)
        # K: Conv1×1 + BN fold（dim → nh_kd）[int8 → int32]
        self.k           = QuantBNConv2d(dim, self.nh_kd, kernel_size=1, weight_bit=weight_bit)
        # K を int8 へ再量子化
        self.qact_k      = QuantAct(activation_bit=act_bit)
        # V: Conv1×1 + BN fold（dim → dh）[int8 → int32]
        self.v           = QuantBNConv2d(dim, self.dh, kernel_size=1, weight_bit=weight_bit)
        # V を int8 へ再量子化
        self.qact_v      = QuantAct(activation_bit=act_bit)
        # v_local: DW-Conv(s=2) + BN fold で V を空間縮小（Q と解像度を合わせる）
        self.v_local = QuantBNConv2d(self.dh, self.dh, kernel_size=3, stride=2, padding=1, groups=self.dh, weight_bit=weight_bit)
        # v_local を int8 へ再量子化
        self.qact_vlocal = QuantAct(activation_bit=act_bit)

        # QK^T の行列積 [int8 × int8 → int32]
        self.qmatmul_qk        = QuantMatMul()
        # softmax 直前に int8 へ再量子化
        self.qact_pre_softmax  = QuantAct(activation_bit=act_bit)
        # 整数近似 softmax
        self.int_softmax       = IntSoftmax(output_bit=act_bit)
        # softmax 後を int8 へ再量子化
        self.qact_post_softmax = QuantAct(activation_bit=act_bit)
        # attn × V の行列積 [int8 × int8 → int32]
        self.qmatmul_attnv     = QuantMatMul()
        # v_local + attn@V の加算後を int8 へ再量子化
        self.qact_merge        = QuantAct(activation_bit=act_bit)

        # 出力 projection: ReLU → Conv1×1 + BN fold（dh → out_dim）
        self.proj_act  = nn.ReLU6(inplace=True)
        self.proj_conv = QuantBNConv2d(self.dh, self.out_dim, kernel_size=1, weight_bit=weight_bit)
        # proj 出力を int8 へ再量子化
        self.qact_out  = QuantAct(activation_bit=act_bit)

        # attention bias（Q×K の位置オフセット; float パラメータ）
        pts_q = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        pts_k = list(itertools.product(range(resolution),       range(resolution)))
        attention_offsets: dict = {}
        idxs: list = []
        for p1 in pts_q:
            for p2 in pts_k:
                sz  = 1
                off = (
                    abs(p1[0] * math.ceil(resolution / self.resolution2) - p2[0] + (sz - 1) / 2),
                    abs(p1[1] * math.ceil(resolution / self.resolution2) - p2[1] + (sz - 1) / 2),
                )
                if off not in attention_offsets:
                    attention_offsets[off] = len(attention_offsets)
                idxs.append(attention_offsets[off])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(self.N2, self.N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        B, C, H, W = x.shape

        # Q 生成（QLGQuery; stride=2 ダウンサンプル付き）[int8]
        q, sf_q = self.q(x, sf)

        # K 生成: Conv1×1 + BN fold（元解像度の x から）[int8 → int32]
        k, sf_k = self.k(x, sf)
        # K を int8 へ再量子化
        k, sf_k = self.qact_k(k, sf_k)

        # V 生成: Conv1×1 + BN fold（元解像度の x から）[int8 → int32]
        v, sf_v = self.v(x, sf)
        # V を int8 へ再量子化
        v, sf_v = self.qact_v(v, sf_v)

        # v_local: DW-Conv(s=2) で V を空間縮小（Q と同じ解像度に揃える）[int8 → int32]
        v_local, sf_vl = self.v_local(v, sf_v)
        # v_local を int8 へ再量子化
        v_local, sf_vl = self.qact_vlocal(v_local, sf_vl)

        # Q: (B, nh_kd, res2, res2) → (B, heads, N2, key_dim)
        q     = q.flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        # K: (B, nh_kd, res, res)   → (B, heads, key_dim, N)（転置済み）
        k     = k.flatten(2).reshape(B, self.num_heads, -1, self.N)
        # V: (B, dh, res, res)       → (B, heads, N, d)
        v_seq = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        # Q @ K^T（QuantMatMul）: int8 × int8 → int32 相当 [B, heads, N2, N]
        attn, sf_attn = self.qmatmul_qk(q, sf_q, k, sf_k)
        # key_dim^-0.5 でスケーリング（float 乗算）
        attn = attn * self.scale
        # attention biases を float のまま加算
        ab = (self.ab if hasattr(self, 'ab') else self.attention_biases[:, self.attention_bias_idxs])
        attn = attn + ab.to(attn.device)
        # softmax 直前に int8 へ再量子化
        attn, sf_attn = self.qact_pre_softmax(attn)
        # IntSoftmax: 整数近似 softmax [int8 → int8]
        attn, sf_attn = self.int_softmax(attn, sf_attn)
        # softmax 後を int8 へ再量子化
        attn, sf_attn = self.qact_post_softmax(attn, sf_attn)

        # attn @ V（QuantMatMul）: int8 × int8 → int32 相当 [B, heads, N2, d]
        out, sf_out = self.qmatmul_attnv(attn, sf_attn, v_seq, sf_v)

        # (B, heads, N2, d) → (B, dh, res2, res2) に形状を戻す
        out = out.transpose(2, 3).reshape(B, self.dh, self.resolution2, self.resolution2)

        # v_local（stride=2 後: res2 解像度）と加算（QuantAct identity で融合）[int8]
        out, sf_out = self.qact_merge(out, sf_out, identity=v_local, identity_scaling_factor=sf_vl)

        # 出力 projection: ReLU → Conv1×1 + BN fold（dh → out_dim）[int8 → int32]
        out = self.proj_act(out)
        out, sf_out = self.proj_conv(out, sf_out)
        # int32 → int8 へ再量子化
        out, sf_out = self.qact_out(out, sf_out)

        return out, sf_out


# ── QEmbeddingAttn ────────────────────────────────────────────────────────────
class QEmbeddingAttn(nn.Module):
    """
    量子化 Embedding（asub=True 版; tiny_eformer Embedding の QAT 版）。
    attention ストリーム（QAttention4DDownsample）と
    conv ストリーム（QuantBNConv2d）を並列に走らせ、加算してからまとめて再量子化する。
    """
    def __init__(self, in_chs: int, out_chs: int, resolution: int,
                 patch_size: int = 3, stride: int = 2, padding: int = 1,
                 weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        # attention ストリーム: Q は stride=2 で縮小、K/V は元解像度から生成
        self.attn = QAttention4DDownsample(dim=in_chs, out_dim=out_chs, resolution=resolution, weight_bit=weight_bit, act_bit=act_bit)
        # conv ストリーム: Conv(patch_size, s=stride) + BN fold
        self.conv = QuantBNConv2d(in_chs, out_chs, kernel_size=patch_size, stride=stride, padding=padding, weight_bit=weight_bit)
        # attn_out + conv_out の加算後を int8 へ再量子化（identity で2スケールを融合）
        self.qact_merge = QuantAct(activation_bit=act_bit)

    def forward(self, x: torch.Tensor, sf: torch.Tensor):
        # attention ストリーム: (x_int8, sf) → (attn_out_int8, sf_attn)
        attn_out, sf_attn = self.attn(x, sf)
        # conv ストリーム: Conv + BN fold [int8 × int8-weight → int32 相当]
        conv_out, sf_conv = self.conv(x, sf)
        # attn_out + conv_out を QuantAct で int8 へ再量子化（identity で2スケールを揃えて加算）
        out, sf_out = self.qact_merge(attn_out, sf_attn, identity=conv_out, identity_scaling_factor=sf_conv)
        return out, sf_out
