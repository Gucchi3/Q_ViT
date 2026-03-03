"""
module.py
i-ViT の量子化モジュールをコピーしたファイル。
QuantAct / QuantLinear / QuantBNConv2d / QuantConv2d など主要クラスを含む。
"""

import math
import decimal
from decimal import Decimal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter


# ══════════════════════════════════════════════════════════════════════════════
# ── 量子化ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def linear_quantize(input, scale, zero_point, is_weight):
    """浮動小数点テンソルを整数に量子化する。"""
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
    return torch.round(1. / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """対称量子化のスケールファクタを計算する。"""
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps
        max_val = torch.max(-min_val, max_val)
        scale = max_val / float(n)
        scale.clamp_(eps)
    return scale


# ── STE 付き量子化関数 ─────────────────────────────────────────────────────────

class SymmetricQuantFunction(Function):
    """対称量子化（STE バックワード対応）。"""

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


class floor_ste(Function):
    """Straight-through Estimator for torch.floor()"""

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """Straight-through Estimator for torch.round()"""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def batch_frexp(inputs, max_bit=31):
    """スケーリングファクタを仮数部と指数部に分解する。"""
    shape_of_input = inputs.size()
    device = inputs.device
    inputs = inputs.view(-1)
    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            Decimal(m * (2 ** max_bit)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP)
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    output_e = float(max_bit) - output_e
    return (
        torch.from_numpy(output_m).to(device).view(shape_of_input),
        torch.from_numpy(output_e).to(device).view(shape_of_input),
    )


class fixedpoint_mul(Function):
    """固定小数点乗算（STE バックワード対応）。"""

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
        return (
            grad_output.clone() / ctx.z_scaling_factor,
            None, None, None, None,
            identity_grad, None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── 量子化レイヤー
# ══════════════════════════════════════════════════════════════════════════════

class QuantAct(nn.Module):
    """
    活性化の量子化レイヤー。

    Parameters
    ----------
    activation_bit : int
        量子化ビット幅。
    act_range_momentum : float
        活性化範囲の移動平均モメンタム。
    running_stat : bool
        True のとき移動平均で範囲を更新する。
    per_channel : bool
        チャネルごと量子化するか（現在は False のみ対応）。
    quant_mode : str
        'symmetric' のみ対応。
    """

    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 quant_mode="symmetric"):
        super().__init__()
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
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return (
            "{0}(activation_bit={1}, quant_mode={2}, "
            "Act_min={3:.2f}, Act_max={4:.2f})"
        ).format(
            self.__class__.__name__, self.activation_bit,
            self.quant_mode, self.min_val.item(), self.max_val.item(),
        )

    def fix(self):
        self.running_stat = False

    def unfix(self):
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
                    self.min_val = (
                        self.min_val * self.act_range_momentum
                        + cur_min * (1 - self.act_range_momentum)
                    )
                    self.max_val = (
                        self.max_val * self.act_range_momentum
                        + cur_max * (1 - self.act_range_momentum)
                    )
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val
            )

        if pre_act_scaling_factor is None:
            quant_act_int = self.act_function(
                x, self.activation_bit, self.act_scaling_factor, False
            )
        else:
            quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor,
                self.activation_bit, self.quant_mode,
                self.act_scaling_factor,
                identity, identity_scaling_factor,
            )

        correct_output_scale = self.act_scaling_factor.view(-1)
        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantLinear(nn.Linear):
    """
    重みを量子化した全結合層。

    Parameters
    ----------
    weight_bit : int
        重みの量子化ビット幅。
    bias_bit : int
        バイアスの量子化ビット幅。
    per_channel : bool
        チャネルごと量子化するか。
    quant_mode : str
        'symmetric' のみ対応。
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric'):
        super().__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (bias_bit is not None)
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super().__repr__()
        return "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode
        )

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
                raise Exception('per_channel=True only.')

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True
        )
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True
            )
        else:
            self.bias_integer = None

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return (
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
            * bias_scaling_factor,
            bias_scaling_factor,
        )


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


class QuantConv2d(nn.Conv2d):
    """
    重みを量子化した畳み込み層（BN なし版）。

    Parameters
    ----------
    weight_bit : int
        重みの量子化ビット幅。
    bias_bit : int
        バイアスの量子化ビット幅。
    quant_mode : str
        'symmetric' のみ対応。
    per_channel : bool
        チャネルごと量子化するか。
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
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,)
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (bias_bit is not None)

        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        s = super().__repr__()
        return "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode
        )

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, pre_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                self.min_val = v.min(axis=1).values
                self.max_val = v.max(axis=1).values
            else:
                raise Exception('per_channel=True only.')

            self.conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.conv_scaling_factor, True)
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        if self.bias is not None:
            self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor, True)
        else:
            self.bias_integer = None

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.stride, self.padding, self.dilation, self.groups) * correct_output_scale, correct_output_scale)
