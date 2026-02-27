"""
quant_module.py
I-ViT 量子化モジュール（量子化ユーティリティ＋量子化レイヤーの統合ファイル）
"""

import pretty_errors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ══════════════════════════════════════════════════════════════════════════════
# ── 量子化ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

# ── 線形量子化関数 ───────────────────────────────────────────────────────────────────
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


# ── 量子化定数 ─────────────────────────────────────────────────────────────────────────
_EPS_F32: float = torch.finfo(torch.float32).eps  # 1.1920929e-07（モジュールロード時1回だけ評価）


# ── 対称量子化スケール計算 ────────────────────────────────────────────────────────────
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
        max_val = torch.max(-min_val, max_val)
        scale = max_val / float(n)
        scale.clamp_(min=_EPS_F32)

    return scale


# ── 対称量子化関数（STE対応） ─────────────────────────────────────────────────────────
class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        scale = specified_scale
        zero_point = x.new_zeros(1)
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


# ── ３値量子化関数（STE対応） ─────────────────────────────────────────────────────────
class TernaryQuantFunction(Function):
    """
    SymmetricQuantFunction と同一構造だが、clamp を固定で [-1, 1] にすることで
    {-1, 0, +1} の３値量子化を行う。STE の backward も SymmetricQuantFunction と同じ。
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        scale = specified_scale
        zero_point = x.new_zeros(1)
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


# ── STE 関数 ───────────────────────────────────────────────────────────────────────
def floor_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.floor(x) - x).detach()


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


# ── 固定小数点乗算（STE対応） ─────────────────────────────────────────────────────────
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

            # z_int ∈ [-128,127]（8bit）、結果も同範囲なので float32 精度で十分。
            # new_scale = pre_sf/z_sf は元の m/2^e と数学的に等価。
            z_int = torch.round(pre_act / pre_act_scaling_factor)
            new_scale = reshape(pre_act_scaling_factor / z_scaling_factor)
            output = torch.round(z_int * new_scale)

            if identity is not None:
                wx_int = torch.round(identity / identity_scaling_factor)
                new_scale1 = reshape(identity_scaling_factor / z_scaling_factor)
                output = output + torch.round(wx_int * new_scale1)

            if bit_num in [4, 8, 16, 32]:
                if quant_mode == 'symmetric':
                    return torch.clamp(output, -n - 1, n)
                else:
                    return torch.clamp(output, 0, n)
            else:
                return output

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

# ── 量子化線形層 ─────────────────────────────────────────────────────────────────────
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
        if self.training:
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
        else:
            bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
               * bias_scaling_factor, bias_scaling_factor


# ── ３値量子化線形層 ──────────────────────────────────────────────────────────────────
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

        if self.training:
            with torch.no_grad():
                w = self.weight
                if self.per_channel:
                    v = w.reshape(w.shape[0], -1)
                    abs_mean = v.abs().mean(axis=1)
                    self.fc_scaling_factor = abs_mean.clamp(min=_EPS_F32)
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
        else:
            bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
               * bias_scaling_factor, bias_scaling_factor


# ── 量子化活性化層 ────────────────────────────────────────────────────────────────────
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
        self._quant_n = 2 ** (activation_bit - 1) - 1  # 量子化レンジ上限（定数）

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
                cur_min = x_act.min()
                cur_max = x_act.max()
                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = self.min_val * self.act_range_momentum + \
                                   cur_min * (1 - self.act_range_momentum)
                    self.max_val = self.max_val * self.act_range_momentum + \
                                   cur_max * (1 - self.act_range_momentum)
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


# ── 量子化行列積層 ────────────────────────────────────────────────────────────────────
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


# ── 量子化畳み込み層 ──────────────────────────────────────────────────────────────────
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
        if self.training:
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
        else:
            bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.stride, self.padding,
                         self.dilation, self.groups) * correct_output_scale, correct_output_scale)


# ── 整数レイヤー正規化 ────────────────────────────────────────────────────────────────
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
        mean_int = round_ste(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        # floor_ste.apply を20回呼ぶ代わりに no_grad + torch.floor でCUDA完結
        with torch.no_grad():
            k = var_int.new_full(var_int.shape, 2**16)
            for _ in range(10):
                k = torch.floor((k + torch.floor(var_int / k)) / 2)
        std_int = k

        factor = floor_ste((2 ** 31 - 1) / std_int)
        y_int = floor_ste(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste(bias / scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor


# ── 整数GELU活性化関数 ────────────────────────────────────────────────────────────────
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
        self.register_buffer('sigmoid_scaling_factor', torch.tensor([1 / 2 ** (output_bit - 1)]))
        self._x0_int_cache = None  # eval時キャッシュ

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste(x_int / 2) - floor_ste(x_int / 2 ** 4)

        with torch.no_grad():
            if self._x0_int_cache is None or self.training:
                x0_int = torch.floor(-1.0 / scaling_factor)
                if not self.training:
                    self._x0_int_cache = x0_int
            else:
                x0_int = self._x0_int_cache
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste(exp_int * 2 ** (self.n - q)), min=0)
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
        factor = floor_ste((2 ** 31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * self.sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


# ── 整数ソフトマックス ────────────────────────────────────────────────────────────────
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
        self.register_buffer('out_scaling_factor', torch.tensor([1 / 2 ** (output_bit - 1)]))
        self._x0_int_cache = None  # eval時キャッシュ

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste(x_int / 2) - floor_ste(x_int / 2 ** 4)

        with torch.no_grad():
            if self._x0_int_cache is None or self.training:
                x0_int = torch.floor(-1.0 / scaling_factor)
                if not self.training:
                    self._x0_int_cache = x0_int
            else:
                x0_int = self._x0_int_cache
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste((2 ** 31 - 1) / exp_int_sum)
        exp_int = floor_ste(exp_int * factor / 2 ** (31 - self.output_bit + 1))

        self.act_scaling_factor = self.out_scaling_factor
        return exp_int * self.out_scaling_factor, self.out_scaling_factor
