"""
test_model.py
テスト用モデル ── モジュールを1つずつ明示的に定義して接続する実験用フレーム

使い方:
  1. TestModel.__init__ に self.xxx = <モジュール> を1つずつ書いていく
  2. TestModel.forward に x, act_sf を渡しながら1行ずつ接続する
  3. config.json の MODEL を "test" にすると main.py から呼ばれる

S0 パラメータ参考:
  embed_dims = [32, 48, 96, 176]
  layers     = [2, 2, 6, 4]       # 各ステージのブロック数
  vit_num    = 2                  # 各ステージ末尾 vit_num 個が AttnFFN
  e_ratios   = {'0':[4,4], '1':[4,4], '2':[4,3,3,3,4,4], '3':[4,3,3,4]}
  resolution = 224 (入力) → stem後: 56 → 28 → 14 → 7


"""

import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    QuantLinear, QuantAct, QuantConv2d, QuantBNConv2d, QuantMatMul, IntSoftmax,
)


# ══════════════════════════════════════════════════════════════════════════════
# ── 共通ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * r.floor_()


# ══════════════════════════════════════════════════════════════════════════════
# ── 単体モジュール群
# ══════════════════════════════════════════════════════════════════════════════

class Q_Stem(nn.Module):
    """
    stem: Conv3x3 BN ReLU x2 (stride=2 ずつ)
    入力 (B, in_chs, H, W) → 出力 (B, out_chs, H/4, W/4)
    """
    def __init__(self, in_chs: int, out_chs: int):
        super().__init__()
        mid = out_chs // 2
        self.conv1  = QuantBNConv2d(in_chs, mid, 3, stride=2, padding=1)
        self.relu1  = nn.ReLU()
        self.quant1 = QuantAct()
        self.conv2  = QuantBNConv2d(mid, out_chs, 3, stride=2, padding=1)
        self.relu2  = nn.ReLU()
        self.quant2 = QuantAct()

    def forward(self, x, act_sf):
        x, _ = self.conv1(x, act_sf)
        x = self.relu1(x)
        x, act_sf = self.quant1(x)
        x, _ = self.conv2(x, act_sf)
        x = self.relu2(x)
        x, act_sf = self.quant2(x)
        return x, act_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_Mlp(nn.Module):
    """
    MLP: fc1+BN+act → [mid+BN+act] → fc2+BN
    mid_conv=True のとき DW-Conv 3x3 の中間層が入る（AttnFFN/FFN で使用）
    """
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=nn.GELU,
                 drop: float = 0., mid_conv: bool = False):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv

        self.fc1    = QuantBNConv2d(in_features, hidden_features, 1)
        self.act1   = act_layer()
        self.quant1 = QuantAct()

        if mid_conv:
            self.mid       = QuantBNConv2d(hidden_features, hidden_features, 3,
                                            stride=1, padding=1, groups=hidden_features)
            self.mid_act   = act_layer()
            self.mid_quant = QuantAct()

        self.drop1  = nn.Dropout(drop)
        self.fc2    = QuantBNConv2d(hidden_features, out_features, 1)
        self.quant2 = QuantAct()
        self.drop2  = nn.Dropout(drop)

    def forward(self, x, act_sf):
        x, _ = self.fc1(x, act_sf)
        x = self.act1(x)
        x, act_sf = self.quant1(x)
        if self.mid_conv:
            x, _ = self.mid(x, act_sf)
            x = self.mid_act(x)
            x, act_sf = self.mid_quant(x)
        x = self.drop1(x)
        x, _ = self.fc2(x, act_sf)
        x, act_sf = self.quant2(x)
        x = self.drop2(x)
        return x, act_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_LGQuery(nn.Module):
    """
    Local-Global Query (Attention4DDownsample 専用)
    DW-Conv(stride=2) + AvgPool(stride=2) → 加算 → proj(Conv1x1+BN)
    ※ DW-Conv は BN なし (原本通り) → QuantConv2d
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dw_conv   = QuantConv2d(in_dim, in_dim, 3, stride=2, padding=1, groups=in_dim)
        self.pool      = nn.AvgPool2d(1, 2, 0)
        self.in_quant  = QuantAct()
        self.proj_conv = QuantBNConv2d(in_dim, out_dim, 1)
        self.out_quant = QuantAct()

    def forward(self, x, act_sf):
        dw_out, _ = self.dw_conv(x, act_sf)
        pool_out   = self.pool(x)
        q, q_sf    = self.in_quant(dw_out + pool_out)
        q, _       = self.proj_conv(q, q_sf)
        q, q_sf    = self.out_quant(q)
        return q, q_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_Attention4D(nn.Module):
    """
    AttnFFN ブロック内の 4D Self-Attention
    stride 設定時: x 全体を stride_conv で縮小後に Q/K/V を生成し、最後に upsample
    talking_head1/2 は QuantConv2d
    """
    def __init__(self, dim: int = 384, key_dim: int = 32, num_heads: int = 8,
                 attn_ratio: int = 4, resolution: int = 7,
                 act_layer=nn.ReLU, stride: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = key_dim ** -0.5
        self.key_dim   = key_dim
        self.d         = int(attn_ratio * key_dim)
        self.dh        = self.d * num_heads
        self.nh_kd     = key_dim * num_heads

        if stride is not None:
            self.resolution   = math.ceil(resolution / stride)
            self.stride_conv  = QuantBNConv2d(dim, dim, 3, stride=stride, padding=1, groups=dim)
            self.stride_quant = QuantAct()
            self.upsample     = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution  = resolution
            self.stride_conv  = None
            self.stride_quant = None
            self.upsample     = None

        self.N = self.resolution ** 2

        self.q_conv  = QuantBNConv2d(dim, self.nh_kd, 1)
        self.q_quant = QuantAct()
        self.k_conv  = QuantBNConv2d(dim, self.nh_kd, 1)
        self.k_quant = QuantAct()
        self.v_conv  = QuantBNConv2d(dim, self.dh, 1)
        self.v_quant = QuantAct()

        self.v_local_conv  = QuantBNConv2d(self.dh, self.dh, 3, stride=1,
                                            padding=1, groups=self.dh)
        self.v_local_quant = QuantAct()

        self.quant_pre_th1 = QuantAct(8)
        self.talking_head1 = QuantConv2d(num_heads, num_heads, 1)
        self.talking_head2 = QuantConv2d(num_heads, num_heads, 1)

        self.matmul_qk    = QuantMatMul()
        self.quant_attn   = QuantAct(8)
        self.int_softmax  = IntSoftmax(16)
        self.quant_postsm = QuantAct(8)
        self.matmul_av    = QuantMatMul()
        self.quant_av_res = QuantAct()

        self.proj_act      = act_layer()
        self.proj_in_quant = QuantAct()
        self.proj_conv     = QuantBNConv2d(self.dh, dim, 1)
        self.proj_quant    = QuantAct()

        pts = list(itertools.product(range(self.resolution), range(self.resolution)))
        offsets: dict = {}
        idxs: list = []
        for p1 in pts:
            for p2 in pts:
                off = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if off not in offsets:
                    offsets[off] = len(offsets)
                idxs.append(offsets[off])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(self.N, self.N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, act_sf):
        B, C, H, W = x.shape

        if self.stride_conv is not None:
            x, _ = self.stride_conv(x, act_sf)
            x, act_sf = self.stride_quant(x)

        q, _ = self.q_conv(x, act_sf);  q, q_sf = self.q_quant(q)
        q = q.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N).permute(0, 1, 3, 2)

        k, _ = self.k_conv(x, act_sf);  k, k_sf = self.k_quant(k)
        k = k.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N)

        v_feat, _ = self.v_conv(x, act_sf);  v_feat, v_sf = self.v_quant(v_feat)
        v_local, _ = self.v_local_conv(v_feat, v_sf);  v_local, _ = self.v_local_quant(v_local)
        v = v_feat.flatten(2).reshape(B, self.num_heads, self.d, self.N).permute(0, 1, 3, 2)

        attn, attn_sf = self.matmul_qk(q, q_sf, k, k_sf)
        attn    = attn    * self.scale
        attn_sf = attn_sf * self.scale
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs]
                       if self.training else self.ab)

        attn, attn_sf    = self.quant_pre_th1(attn, attn_sf)
        attn, attn_sf    = self.talking_head1(attn, attn_sf)
        attn, attn_q_sf  = self.quant_attn(attn, attn_sf)
        attn, attn_q_sf  = self.int_softmax(attn, attn_q_sf)
        attn, attn_q_sf  = self.talking_head2(attn, attn_q_sf)
        attn, attn_q_sf  = self.quant_postsm(attn, attn_q_sf)

        out, out_sf = self.matmul_av(attn, attn_q_sf, v, v_sf)
        out = out.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution)
        out = out + v_local
        if self.upsample is not None:
            out = self.upsample(out)
        out, out_sf = self.quant_av_res(out)

        out = self.proj_act(out);  out, out_sf = self.proj_in_quant(out)
        out, _ = self.proj_conv(out, out_sf);  out, out_sf = self.proj_quant(out)
        return out, out_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_Attention4DDownsample(nn.Module):
    """
    Embedding(asub=True) 専用のダウンサンプリング付き 4D Attention
    Q は Q_LGQuery (stride=2)、K/V は元解像度の x から生成
    talking_head なし (原本通り)
    """
    def __init__(self, dim: int = 384, key_dim: int = 16, num_heads: int = 8,
                 attn_ratio: int = 4, resolution: int = 7,
                 out_dim: int = None, act_layer=nn.ReLU):
        super().__init__()
        self.num_heads   = num_heads
        self.scale       = key_dim ** -0.5
        self.key_dim     = key_dim
        self.d           = int(attn_ratio * key_dim)
        self.dh          = self.d * num_heads
        self.nh_kd       = key_dim * num_heads
        self.out_dim     = out_dim if out_dim is not None else dim
        self.resolution  = resolution
        self.resolution2 = math.ceil(resolution / 2)
        self.N           = resolution ** 2
        self.N2          = self.resolution2 ** 2

        self.q = Q_LGQuery(dim, self.nh_kd)

        self.k_conv  = QuantBNConv2d(dim, self.nh_kd, 1)
        self.k_quant = QuantAct()
        self.v_conv  = QuantBNConv2d(dim, self.dh, 1)
        self.v_quant = QuantAct()

        self.v_local_conv  = QuantBNConv2d(self.dh, self.dh, 3, stride=2,
                                            padding=1, groups=self.dh)
        self.v_local_quant = QuantAct()

        self.matmul_qk    = QuantMatMul()
        self.quant_attn   = QuantAct(8)
        self.int_softmax  = IntSoftmax(16)
        self.matmul_av    = QuantMatMul()
        self.quant_av_res = QuantAct()

        self.proj_act      = act_layer()
        self.proj_in_quant = QuantAct()
        self.proj_conv     = QuantBNConv2d(self.dh, self.out_dim, 1)
        self.proj_quant    = QuantAct()

        pts_q = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        pts_k = list(itertools.product(range(resolution), range(resolution)))
        offsets: dict = {}
        idxs: list = []
        for p1 in pts_q:
            for p2 in pts_k:
                off = (
                    abs(p1[0] * math.ceil(resolution / self.resolution2) - p2[0] + 0.0),
                    abs(p1[1] * math.ceil(resolution / self.resolution2) - p2[1] + 0.0),
                )
                if off not in offsets:
                    offsets[off] = len(offsets)
                idxs.append(offsets[off])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(self.N2, self.N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, act_sf):
        B, C, H, W = x.shape

        q, q_sf = self.q(x, act_sf)
        q = q.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N2).permute(0, 1, 3, 2)

        k, _ = self.k_conv(x, act_sf);  k, k_sf = self.k_quant(k)
        k = k.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N)

        v_feat, _ = self.v_conv(x, act_sf);  v_feat, v_sf = self.v_quant(v_feat)
        v_local, _ = self.v_local_conv(v_feat, v_sf);  v_local, _ = self.v_local_quant(v_local)
        v = v_feat.flatten(2).reshape(B, self.num_heads, self.d, self.N).permute(0, 1, 3, 2)

        attn, attn_sf = self.matmul_qk(q, q_sf, k, k_sf)
        attn    = attn    * self.scale
        attn_sf = attn_sf * self.scale
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs]
                       if self.training else self.ab)

        attn, attn_q_sf = self.quant_attn(attn, attn_sf)
        attn, attn_q_sf = self.int_softmax(attn, attn_q_sf)

        out, out_sf = self.matmul_av(attn, attn_q_sf, v, v_sf)
        out = out.transpose(2, 3).reshape(B, self.dh, self.resolution2, self.resolution2)
        out = out + v_local
        out, out_sf = self.quant_av_res(out)

        out = self.proj_act(out);  out, out_sf = self.proj_in_quant(out)
        out, _ = self.proj_conv(out, out_sf);  out, out_sf = self.proj_quant(out)
        return out, out_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_FFN(nn.Module):
    """
    FFN ブロック: x + drop_path(layer_scale * mlp(x))
    mid_conv=True の Q_Mlp を使用
    """
    def __init__(self, dim: int, mlp_ratio: float = 4., act_layer=nn.GELU,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init_value: float = 1e-5):
        super().__init__()
        self.mlp = Q_Mlp(dim, hidden_features=int(dim * mlp_ratio),
                          act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
        self.quant_res = QuantAct()

    def forward(self, x, act_sf):
        residual = x
        mlp_out, _ = self.mlp(x, act_sf)
        if self.use_layer_scale:
            mlp_out = self.drop_path(self.layer_scale_2 * mlp_out)
        else:
            mlp_out = self.drop_path(mlp_out)
        x, act_sf = self.quant_res(residual + mlp_out)
        return x, act_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_AttnFFN(nn.Module):
    """
    AttnFFN ブロック: token_mixer(Q_Attention4D) + mlp(Q_Mlp)
    x → attn残差 → quant_res1 → mlp残差 → quant_res2
    """
    def __init__(self, dim: int, mlp_ratio: float = 4., act_layer=nn.ReLU,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init_value: float = 1e-5,
                 resolution: int = 7, stride: int = None):
        super().__init__()
        self.token_mixer = Q_Attention4D(dim, resolution=resolution,
                                          act_layer=act_layer, stride=stride)
        self.mlp = Q_Mlp(dim, hidden_features=int(dim * mlp_ratio),
                          act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
        self.quant_res1 = QuantAct()
        self.quant_res2 = QuantAct()

    def forward(self, x, act_sf):
        res1 = x
        attn_out, _ = self.token_mixer(x, act_sf)
        if self.use_layer_scale:
            attn_out = self.drop_path(self.layer_scale_1 * attn_out)
        else:
            attn_out = self.drop_path(attn_out)
        x, act_sf = self.quant_res1(res1 + attn_out)

        res2 = x
        mlp_out, _ = self.mlp(x, act_sf)
        if self.use_layer_scale:
            mlp_out = self.drop_path(self.layer_scale_2 * mlp_out)
        else:
            mlp_out = self.drop_path(mlp_out)
        x, act_sf = self.quant_res2(res2 + mlp_out)
        return x, act_sf


# ─────────────────────────────────────────────────────────────────────────────
class Q_Embedding(nn.Module):
    """
    Embedding (ステージ間ダウンサンプリング)
      asub=False: Conv+BN → QuantAct
      asub=True : Q_Attention4DDownsample + QuantBNConv2d を並列加算 → QuantAct
    """
    def __init__(self, in_chs: int, out_chs: int,
                 patch_size: int = 3, stride: int = 2, padding: int = 1,
                 asub: bool = False, resolution: int = None,
                 act_layer=nn.ReLU):
        super().__init__()
        self.asub = asub
        if asub:
            self.attn       = Q_Attention4DDownsample(
                                  dim=in_chs, out_dim=out_chs,
                                  resolution=resolution, act_layer=act_layer)
            self.conv       = QuantBNConv2d(in_chs, out_chs, patch_size,
                                             stride=stride, padding=padding)
            self.conv_quant = QuantAct()
            self.quant      = QuantAct()
        else:
            self.conv  = QuantBNConv2d(in_chs, out_chs, patch_size,
                                        stride=stride, padding=padding)
            self.quant = QuantAct()

    def forward(self, x, act_sf):
        if self.asub:
            attn_out, _ = self.attn(x, act_sf)
            conv_out, _  = self.conv(x, act_sf)
            conv_out, _  = self.conv_quant(conv_out)
            x, act_sf    = self.quant(attn_out.float() + conv_out.float())
        else:
            x, _ = self.conv(x, act_sf)
            x, act_sf = self.quant(x)
        return x, act_sf


# ══════════════════════════════════════════════════════════════════════════════
# ── TestModel ── 
# ══════════════════════════════════════════════════════════════════════════════

class TestModel(nn.Module):
    """
    実験用モデル。
    __init__ に self.xxx = <モジュール> を1つずつ記述し、
    forward で (x, act_sf) を渡しながら接続する。

    利用できるモジュール:
        Q_Stem, Q_FFN, Q_AttnFFN, Q_Embedding,
        Q_Attention4D, Q_Attention4DDownsample,
        Q_LGQuery, Q_Mlp, DropPath,
        QuantAct, QuantBNConv2d, QuantConv2d,
        QuantMatMul, QuantLinear, IntSoftmax
    """

    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__()
        # ── ここにブロックを1つずつ定義していく ──────────────────────────

        pass

    def forward(self, x):
        # ── ここに実行順を1行ずつ書いていく ─────────────────────────────

        pass


# ── ファクトリ関数 ────────────────────────────────────────────────────────────

def test(pretrained: bool = False, num_classes: int = 10, **kwargs):
    """config の MODEL="test" で呼ばれるエントリポイント"""
    model = TestModel(num_classes=num_classes, **kwargs)
    return model
