"""
q_eformer_v2.py  -- 量子化 EfficientFormerV2 (QAT)

元の EfficientFormerV2 の構造を忠実に量子化:
  stem       : Conv3x3+BN+ReLU x2  -> QuantBNConv2d + QuantAct
  Attention4D: stride 時は x 全体を stride_conv で縮小してから Q/K/V を生成
               (原実装と同じく Q/K/V すべて縮小後 x から)
  Attention4DDownsample: asub Embedding 専用。
               Q は LGQuery (DW-Conv stride=2 + AvgPool -> proj)
               K/V は元解像度の x から
  Embedding  : normal->QuantBNConv2d, asub->attn+conv 並列加算->QuantAct
  FFN/AttnFFN: layer_scale 後 QuantAct で残差加算
  QuantAct は必ずメンバーとして保持(forward 内で new しない)
"""

import os
import math
import itertools
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    QuantLinear, QuantAct, QuantConv2d, QuantBNConv2d, QuantMatMul, IntSoftmax,
)
from .utils import to_2tuple, trunc_normal_

__all__ = [
    'Q_EfficientFormerV2',
    'q_eformer_v2_s0', 'q_eformer_v2_s1', 'q_eformer_v2_s2', 'q_eformer_v2_l',
]

# ─── サイズ設定 ───────────────────────────────────────────────────────────────────
EfficientFormer_width = {
    'L':  [40,  80,  192, 384],
    'S2': [32,  64,  144, 288],
    'S1': [32,  48,  120, 224],
    'S0': [32,  48,   96, 176],
}
EfficientFormer_depth = {
    'L':  [5,  5,  15, 10],
    'S2': [4,  4,  12,  8],
    'S1': [3,  3,   9,  6],
    'S0': [2,  2,   6,  4],
}
expansion_ratios_L  = {'0':[4,4,4,4,4],'1':[4,4,4,4,4],
                       '2':[4,4,4,4,3,3,3,3,3,3,3,4,4,4,4],'3':[4,4,4,3,3,3,3,4,4,4]}
expansion_ratios_S2 = {'0':[4,4,4,4],'1':[4,4,4,4],
                       '2':[4,4,3,3,3,3,3,3,4,4,4,4],'3':[4,4,3,3,3,3,4,4]}
expansion_ratios_S1 = {'0':[4,4,4],'1':[4,4,4],
                       '2':[4,4,3,3,3,3,4,4,4],'3':[4,4,3,3,4,4]}
expansion_ratios_S0 = {'0':[4,4],'1':[4,4],
                       '2':[4,3,3,3,4,4],'3':[4,3,3,4]}

PRETRAINED_WEIGHTS = {
    'S0': './data/eformer_s0_450.pth',
    'S1': './data/eformer_s1_450.pth',
    'S2': './data/eformer_s2_450.pth',
    'L':  './data/eformer_l_450.pth',
}


# ─── DropPath ─────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
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


# ─── Q_StageSequential ────────────────────────────────────────────────────────────
class Q_StageSequential(nn.Module):
    """(x, act_scaling_factor) を保ちながらブロックを順次実行"""
    def __init__(self, blocks: list):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, act_sf):
        for blk in self.blocks:
            x, act_sf = blk(x, act_sf)
        return x, act_sf


# ─── Q_Stem ───────────────────────────────────────────────────────────────────────
class Q_Stem(nn.Module):
    """stem: Conv3x3 BN ReLU x2 (stride=2 each)"""
    def __init__(self, in_chs, out_chs):
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


# ─── Q_Mlp ────────────────────────────────────────────────────────────────────────
class Q_Mlp(nn.Module):
    """fc1+BN+act -> [mid+BN+act] -> fc2+BN。QuantBNConv2d + QuantAct x n。"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., mid_conv=False):
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


# ─── Q_LGQuery ────────────────────────────────────────────────────────────────────
class Q_LGQuery(nn.Module):
    """
    元: local(DW-Conv stride=2) + pool(AvgPool stride=2) -> proj(Conv1x1+BN)
    DW-Conv は BN なし (原実装と同じ) -> QuantConv2d。
    pool との加算後 fresh QuantAct -> proj_conv(QuantBNConv2d) -> QuantAct。
    """
    def __init__(self, in_dim, out_dim):
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


# ─── Q_Attention4D ────────────────────────────────────────────────────────────────
class Q_Attention4D(nn.Module):
    """
    AttnFFN ブロック内の 4D アテンション。原実装 Attention4D に対応。
    stride 設定時: stride_conv を x 全体に適用し、Q/K/V はすべて縮小後の x から生成。
    出力は upsample で元解像度に戻す。
    位置バイアスは (N x N) 正方形。
    """
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4, resolution=7, act_layer=nn.ReLU, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = key_dim ** -0.5
        self.key_dim   = key_dim
        self.d         = int(attn_ratio * key_dim)
        self.dh        = self.d * num_heads
        self.nh_kd     = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv  = QuantBNConv2d(dim, dim, 3, stride=stride, padding=1, groups=dim)
            self.stride_quant = QuantAct()
            self.upsample     = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution  = resolution
            self.stride_conv  = None
            self.stride_quant = None
            self.upsample     = None

        self.N = self.resolution ** 2   # 正方形 (原実装 N = N2)

        self.q_conv  = QuantBNConv2d(dim, self.nh_kd, 1)
        self.q_quant = QuantAct()
        self.k_conv  = QuantBNConv2d(dim, self.nh_kd, 1)
        self.k_quant = QuantAct()
        self.v_conv  = QuantBNConv2d(dim, self.dh, 1)
        self.v_quant = QuantAct()

        self.v_local_conv  = QuantBNConv2d(self.dh, self.dh, 3, stride=1,
                                            padding=1, groups=self.dh)
        self.v_local_quant = QuantAct()

        self.quant_pre_th1 = QuantAct(8)       # talking_head1 前
        self.talking_head1 = QuantConv2d(num_heads, num_heads, 1)
        self.talking_head2 = QuantConv2d(num_heads, num_heads, 1)

        self.matmul_qk     = QuantMatMul()
        self.quant_attn    = QuantAct(8)       # talking_head1 後, softmax 前
        self.int_softmax   = IntSoftmax(16)    # 整数近似 softmax
        self.quant_postsm  = QuantAct(8)       # talking_head2 後
        self.matmul_av     = QuantMatMul()
        self.quant_av_res  = QuantAct()        # out + v_local 加算後 fresh quant

        self.proj_act      = act_layer()
        self.proj_in_quant = QuantAct()
        self.proj_conv     = QuantBNConv2d(self.dh, dim, 1)
        self.proj_quant    = QuantAct()

        pts = list(itertools.product(range(self.resolution), range(self.resolution)))
        offsets: dict = {}
        idxs: list = []
        for p1 in pts:
            for p2 in pts:
                off = (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
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

        # ── stride_conv: x 全体を縮小（Q/K/V はすべて縮小後 x から生成）────
        if self.stride_conv is not None:
            x, _ = self.stride_conv(x, act_sf)
            # stride_conv 後 fresh quant（per-ch scale → scalar scale 変換）
            x, act_sf = self.stride_quant(x)

        # ── Q, K, V 生成 ────────────────────────────────────────────────
        q, _ = self.q_conv(x, act_sf)
        q, q_sf = self.q_quant(q)
        # (B, nh_kd, H', W') → (B, num_heads, N, key_dim)
        q = q.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N).permute(0, 1, 3, 2)

        k, _ = self.k_conv(x, act_sf)
        k, k_sf = self.k_quant(k)
        # (B, nh_kd, H', W') → (B, num_heads, key_dim, N)
        k = k.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N)

        v_feat, _ = self.v_conv(x, act_sf)
        v_feat, v_sf = self.v_quant(v_feat)

        # v_local: DW-Conv3x3 → (B, dh, H', W')
        v_local, _ = self.v_local_conv(v_feat, v_sf)
        v_local, _ = self.v_local_quant(v_local)

        # (B, dh, H', W') → (B, num_heads, N, d)
        v = v_feat.flatten(2).reshape(B, self.num_heads, self.d, self.N).permute(0, 1, 3, 2)

        # ── Attention score ─────────────────────────────────────────────
        # QuantMatMul は (q_int @ k_int) * (q_sf * k_sf) の float を返す
        # ★ attn はすでに float 表現 → attn_sf を掛けると double-scaling になる
        attn, attn_sf = self.matmul_qk(q, q_sf, k, k_sf)

        # scale 適用（float 上で一度だけ）
        attn    = attn * self.scale
        attn_sf = attn_sf * self.scale

        # 位置バイアス加算
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs]
                       if self.training else self.ab)

        # talking_head1 前の量子化（scale を scalar に整理）
        attn, attn_sf = self.quant_pre_th1(attn, attn_sf)

        # talking_head1（QuantConv2d）
        attn, attn_sf = self.talking_head1(attn, attn_sf)

        # QuantAct: attn_sf を pre_act_sf として渡し fixedpoint_mul で量子化
        attn, attn_q_sf = self.quant_attn(attn, attn_sf)

        # IntSoftmax: 整数近似 softmax
        attn, attn_q_sf = self.int_softmax(attn, attn_q_sf)

        # talking_head2（QuantConv2d）
        attn, attn_q_sf = self.talking_head2(attn, attn_q_sf)

        # talking_head2 後に再量子化
        attn, attn_q_sf = self.quant_postsm(attn, attn_q_sf)

        # ── Attn @ V ────────────────────────────────────────────────────
        # (B, heads, N, N) @ (B, heads, N, d) → (B, heads, N, d)
        out, out_sf = self.matmul_av(attn, attn_q_sf, v, v_sf)
        # (B, heads, d, N) → (B, dh, res, res)
        out = out.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution)

        # v_local 加算（★ both are already float, do NOT multiply by scale again）
        out = out + v_local

        # upsample（stride あり時のみ）
        if self.upsample is not None:
            out = self.upsample(out)

        # out + v_local 後の fresh quant
        out, out_sf = self.quant_av_res(out)

        # ── proj ─────────────────────────────────────────────────────────
        out = self.proj_act(out)
        out, out_sf = self.proj_in_quant(out)
        out, _ = self.proj_conv(out, out_sf)
        out, out_sf = self.proj_quant(out)
        return out, out_sf


# ─── Q_Attention4DDownsample ──────────────────────────────────────────────────────
class Q_Attention4DDownsample(nn.Module):
    """
    Embedding(asub=True) 専用。原実装 Attention4DDownsample に対応。
      Q: LGQuery (DW-Conv stride=2 + AvgPool -> proj) -> resolution2 (N2 = resolution2^2)
      K: Conv1x1+BN on x -> resolution  (N = resolution^2)
      V: Conv1x1+BN on x -> resolution
      v_local: DW-Conv stride=2 on v -> resolution2
      attn: Q@K (N2 x N), softmax, attn@V -> reshape (N2) -> + v_local
      proj: act -> Conv1x1+BN -> out_dim
      talking_head なし (原実装と同じ)
    """
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4, resolution=7, out_dim=None, act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = key_dim ** -0.5
        self.key_dim   = key_dim
        self.d         = int(attn_ratio * key_dim)
        self.dh        = self.d * num_heads
        self.nh_kd     = key_dim * num_heads
        self.out_dim   = out_dim if out_dim is not None else dim

        self.resolution  = resolution
        self.resolution2 = math.ceil(resolution / 2)
        self.N  = resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.q = Q_LGQuery(dim, self.nh_kd)

        self.k_conv  = QuantBNConv2d(dim, self.nh_kd, 1)
        self.k_quant = QuantAct()
        self.v_conv  = QuantBNConv2d(dim, self.dh, 1)
        self.v_quant = QuantAct()

        self.v_local_conv  = QuantBNConv2d(self.dh, self.dh, 3, stride=2,
                                            padding=1, groups=self.dh)
        self.v_local_quant = QuantAct()

        self.matmul_qk   = QuantMatMul()
        self.quant_attn  = QuantAct(8)        # softmax 前
        self.int_softmax = IntSoftmax(16)     # 整数近似 softmax
        self.matmul_av   = QuantMatMul()
        self.quant_av_res = QuantAct()        # out + v_local 後 fresh quant

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
                s = 1
                off = (
                    abs(p1[0] * math.ceil(resolution / self.resolution2) - p2[0] + (s-1)/2),
                    abs(p1[1] * math.ceil(resolution / self.resolution2) - p2[1] + (s-1)/2),
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

        k, _ = self.k_conv(x, act_sf)
        k, k_sf = self.k_quant(k)
        k = k.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N)

        v_feat, _ = self.v_conv(x, act_sf)
        v_feat, v_sf = self.v_quant(v_feat)

        v_local, _ = self.v_local_conv(v_feat, v_sf)
        v_local, v_local_sf = self.v_local_quant(v_local)

        v = v_feat.flatten(2).reshape(B, self.num_heads, self.d, self.N).permute(0, 1, 3, 2)

        # ★ QuantMatMul は float を返す。attn_sf を掛けると double-scaling になる
        attn, attn_sf = self.matmul_qk(q, q_sf, k, k_sf)

        # scale 適用（float 上で一度だけ）＋位置バイアス加算
        attn    = attn * self.scale
        attn_sf = attn_sf * self.scale
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs]
                       if self.training else self.ab)

        # QuantAct: attn_sf を pre_act_sf として渡し fixedpoint_mul で量子化
        attn, attn_q_sf = self.quant_attn(attn, attn_sf)

        # IntSoftmax: 整数近似 softmax
        attn, attn_q_sf = self.int_softmax(attn, attn_q_sf)

        # (B, heads, N2, N) @ (B, heads, N, d) → (B, heads, N2, d)
        out, out_sf = self.matmul_av(attn, attn_q_sf, v, v_sf)
        # (B, heads, d, N2) → (B, dh, res2, res2)
        out = out.transpose(2, 3).reshape(B, self.dh, self.resolution2, self.resolution2)

        # v_local 加算（★ both are already float, do NOT multiply by scale again）
        out = out + v_local

        # fresh quant
        out, out_sf = self.quant_av_res(out)

        out = self.proj_act(out)
        out, out_sf = self.proj_in_quant(out)
        out, _ = self.proj_conv(out, out_sf)
        out, out_sf = self.proj_quant(out)
        return out, out_sf


# ─── Q_FFN ────────────────────────────────────────────────────────────────────────
class Q_FFN(nn.Module):
    """元の FFN に対応。x + drop_path(layer_scale * mlp(x))"""
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
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
        # residual も mlp_out もすでに float 表現 → .float() 不要
        x, act_sf = self.quant_res(residual + mlp_out)
        return x, act_sf


# ─── Q_AttnFFN ────────────────────────────────────────────────────────────────────
class Q_AttnFFN(nn.Module):
    """元の AttnFFN に対応。token_mixer(Attention4D) + mlp。"""
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.ReLU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Q_Attention4D(dim, resolution=resolution,
                                         act_layer=act_layer, stride=stride)
        # 原実装に合わせて mlp も同じ act_layer を使用
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
        # res1 も attn_out もすでに float 表現 → .float() 不要
        x, act_sf = self.quant_res1(res1 + attn_out)

        res2 = x
        mlp_out, _ = self.mlp(x, act_sf)
        if self.use_layer_scale:
            mlp_out = self.drop_path(self.layer_scale_2 * mlp_out)
        else:
            mlp_out = self.drop_path(mlp_out)
        x, act_sf = self.quant_res2(res2 + mlp_out)
        return x, act_sf


# ─── Q_Embedding ──────────────────────────────────────────────────────────────────
class Q_Embedding(nn.Module):
    """
    元の Embedding に対応。
      normal (asub=False):  Conv+BN -> QuantAct
      asub=True:  Q_Attention4DDownsample(attn) + QuantBNConv2d(conv) -> 加算 -> QuantAct
                  (元: out = attn(x) + bn(conv(x)))
    """
    def __init__(self, in_chs, out_chs, patch_size=3, stride=2, padding=1,
                 asub=False, resolution=None, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d):
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


# ─── ステージ構築ヘルパー ─────────────────────────────────────────────────────────
def q_eformer_block(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU,
                     drop_rate=0., drop_path_rate=0.,
                     use_layer_scale=True, layer_scale_init_value=1e-5,
                     vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        ratio = e_ratios[str(index)][block_idx] if e_ratios else mlp_ratio
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            stride = 2 if index == 2 else None
            blocks.append(Q_AttnFFN(
                dim, mlp_ratio=ratio, act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution, stride=stride,
            ))
        else:
            blocks.append(Q_FFN(
                dim, mlp_ratio=ratio, act_layer=nn.GELU,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    return blocks


# ─── Q_EfficientFormerV2 ──────────────────────────────────────────────────────────
class Q_EfficientFormerV2(nn.Module):
    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None,
                 pool_size=3, act_layer=nn.GELU, num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 vit_num=0, resolution=224, e_ratios=expansion_ratios_L,
                 in_chans=3, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.quant_input = QuantAct()
        self.patch_embed  = Q_Stem(in_chans, embed_dims[0])

        downsamples = downsamples or [True] * len(layers)
        network_layers = []
        for i in range(len(layers)):
            stage_res = math.ceil(resolution / (2 ** (i + 2)))
            blks = q_eformer_block(
                embed_dims[i], i, layers,
                mlp_ratio=mlp_ratios, act_layer=act_layer,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=stage_res, vit_num=vit_num, e_ratios=e_ratios,
            )
            network_layers.append(Q_StageSequential(blks))
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                asub = (i >= 2)
                network_layers.append(Q_Embedding(
                    embed_dims[i], embed_dims[i + 1],
                    patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                    asub=asub, resolution=stage_res, act_layer=act_layer,
                ))
        self.network = nn.ModuleList(network_layers)

        self.norm       = nn.BatchNorm2d(embed_dims[-1])
        self.quant_norm = QuantAct()
        self.head       = QuantLinear(embed_dims[-1], num_classes)
        self.quant_head = QuantAct()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, act_sf = self.quant_input(x)
        x, act_sf = self.patch_embed(x, act_sf)
        for layer in self.network:
            x, act_sf = layer(x, act_sf)
        x = self.norm(x)
        x, act_sf = self.quant_norm(x)
        x = x.flatten(2).mean(-1)
        x, act_sf = self.head(x, act_sf)
        x, _ = self.quant_head(x, act_sf)
        return x


# ─── pretrained 重みローダー ──────────────────────────────────────────────────────
def _remap_keys_v2(src: dict) -> dict:
    """
    原実装 EfficientFormerV2 の state_dict キーを Q モデルにマッピング。

    Sequential(Conv, BN) -> QuantBNConv2d の変換:
      prefix.0.weight -> target.weight
      prefix.0.bias   -> None (スキップ)
      prefix.1.*      -> target.bn.*
    """

    def _seq2qbn(rest, seq_prefix, target):
        """seq_prefix.0.weight -> target.weight, seq_prefix.1.* -> target.bn.*"""
        if rest == f'{seq_prefix}.0.weight': return f'{target}.weight'
        if rest == f'{seq_prefix}.0.bias':   return None
        m = re.match(rf'^{re.escape(seq_prefix)}\.1\.(.+)$', rest)
        if m: return f'{target}.bn.{m.group(1)}'
        return rest  # 変換対象外 (そのまま返す)

    def _map_attn4d(rest: str):
        """Attention4D (token_mixer) 内キー変換"""
        # q/k/v: Sequential(Conv1x1, BN) -> q_conv/k_conv/v_conv (QuantBNConv2d)
        #   原本キー: {ch}.0.weight / {ch}.0.bias / {ch}.1.*
        for ch in ('q', 'k', 'v'):
            if rest.startswith(f'{ch}.'):
                inner = rest[len(f'{ch}.'):]
                if inner == '0.weight': return f'{ch}_conv.weight'
                if inner == '0.bias':   return None
                m2 = re.match(r'^1\.(.+)$', inner)
                if m2: return f'{ch}_conv.bn.{m2.group(1)}'
        # v_local
        if rest == 'v_local.0.weight': return 'v_local_conv.weight'
        if rest == 'v_local.0.bias':   return None
        m = re.match(r'^v_local\.1\.(.+)$', rest)
        if m: return f'v_local_conv.bn.{m.group(1)}'
        # stride_conv: Sequential(Conv3x3, BN)
        if rest == 'stride_conv.0.weight': return 'stride_conv.weight'
        if rest == 'stride_conv.0.bias':   return None
        m = re.match(r'^stride_conv\.1\.(.+)$', rest)
        if m: return f'stride_conv.bn.{m.group(1)}'
        # proj: Sequential(act=0, Conv=1, BN=2)
        if rest == 'proj.1.weight': return 'proj_conv.weight'
        if rest == 'proj.1.bias':   return None
        m = re.match(r'^proj\.2\.(.+)$', rest)
        if m: return f'proj_conv.bn.{m.group(1)}'
        return rest

    def _map_attn4d_ds(rest: str):
        """Attention4DDownsample (asub Embedding の attn) 内キー変換"""
        # q.local.0 (Conv, BN なし) -> q.dw_conv
        if rest == 'q.local.0.weight': return 'q.dw_conv.weight'
        if rest == 'q.local.0.bias':   return None
        # q.pool: パラメータなし
        # q.proj: Sequential(Conv1x1, BN) -> q.proj_conv
        if rest == 'q.proj.0.weight': return 'q.proj_conv.weight'
        if rest == 'q.proj.0.bias':   return None
        m = re.match(r'^q\.proj\.1\.(.+)$', rest)
        if m: return f'q.proj_conv.bn.{m.group(1)}'
        # k: Sequential(Conv1x1, BN) -> k_conv
        if rest == 'k.0.weight': return 'k_conv.weight'
        if rest == 'k.0.bias':   return None
        m = re.match(r'^k\.1\.(.+)$', rest)
        if m: return f'k_conv.bn.{m.group(1)}'
        # v: Sequential(Conv1x1, BN) -> v_conv
        if rest == 'v.0.weight': return 'v_conv.weight'
        if rest == 'v.0.bias':   return None
        m = re.match(r'^v\.1\.(.+)$', rest)
        if m: return f'v_conv.bn.{m.group(1)}'
        # v_local: Sequential(Conv3x3 stride=2, BN) -> v_local_conv
        if rest == 'v_local.0.weight': return 'v_local_conv.weight'
        if rest == 'v_local.0.bias':   return None
        m = re.match(r'^v_local\.1\.(.+)$', rest)
        if m: return f'v_local_conv.bn.{m.group(1)}'
        # proj: Sequential(act=0, Conv=1, BN=2) -> proj_conv
        if rest == 'proj.1.weight': return 'proj_conv.weight'
        if rest == 'proj.1.bias':   return None
        m = re.match(r'^proj\.2\.(.+)$', rest)
        if m: return f'proj_conv.bn.{m.group(1)}'
        return rest

    def _map_block(rest: str):
        """FFN / AttnFFN ブロック内キー変換"""
        rest = re.sub(r'^mlp\.norm1\.', 'mlp.fc1.bn.', rest)
        rest = re.sub(r'^mlp\.norm2\.', 'mlp.fc2.bn.', rest)
        rest = re.sub(r'^mlp\.mid_norm\.', 'mlp.mid.bn.', rest)
        if re.fullmatch(r'mlp\.(fc1|fc2|mid)\.bias', rest): return None
        if rest.startswith('token_mixer.'):
            inner = rest[len('token_mixer.'):]
            r = _map_attn4d(inner)
            return None if r is None else f'token_mixer.{r}'
        return rest

    def _map(k: str):
        if k.startswith('dist_head.'): return None

        # patch_embed (stem): 0=Conv1, 1=BN1, 2=act(param なし), 3=Conv2, 4=BN2, 5=act
        if k == 'patch_embed.0.weight': return 'patch_embed.conv1.weight'
        if k == 'patch_embed.0.bias':   return None
        m = re.match(r'^patch_embed\.1\.(.+)$', k)
        if m: return f'patch_embed.conv1.bn.{m.group(1)}'
        if k == 'patch_embed.3.weight': return 'patch_embed.conv2.weight'
        if k == 'patch_embed.3.bias':   return None
        m = re.match(r'^patch_embed\.4\.(.+)$', k)
        if m: return f'patch_embed.conv2.bn.{m.group(1)}'

        # stage blocks: network.N.M.* -> network.N.blocks.M.*
        m = re.match(r'^(network\.\d+)\.(\d+)\.(.+)$', k)
        if m:
            prefix, blk, rest = m.group(1), m.group(2), m.group(3)
            r = _map_block(rest)
            return None if r is None else f'{prefix}.blocks.{blk}.{r}'

        # Embedding asub: attn.*
        m = re.match(r'^(network\.\d+)\.attn\.(.+)$', k)
        if m:
            prefix, rest = m.group(1), m.group(2)
            r = _map_attn4d_ds(rest)
            return None if r is None else f'{prefix}.attn.{r}'

        # Embedding asub: conv + bn -> conv (QuantBNConv2d)
        m = re.match(r'^(network\.\d+)\.conv\.weight$', k)
        if m: return f'{m.group(1)}.conv.weight'
        m = re.match(r'^(network\.\d+)\.conv\.bias$', k)
        if m: return None
        m = re.match(r'^(network\.\d+)\.bn\.(.+)$', k)
        if m: return f'{m.group(1)}.conv.bn.{m.group(2)}'

        # Embedding normal: proj + norm -> conv (QuantBNConv2d)
        m = re.match(r'^(network\.\d+)\.proj\.weight$', k)
        if m: return f'{m.group(1)}.conv.weight'
        m = re.match(r'^(network\.\d+)\.proj\.bias$', k)
        if m: return None
        m = re.match(r'^(network\.\d+)\.norm\.(.+)$', k)
        if m: return f'{m.group(1)}.conv.bn.{m.group(2)}'

        return k

    return {nk: v for k, v in src.items() if (nk := _map(k)) is not None}


def _load_pretrained(model: Q_EfficientFormerV2, variant: str):
    weight_path = PRETRAINED_WEIGHTS.get(variant, '')
    if not weight_path or not os.path.exists(weight_path):
        print(f'[Warning] q_eformer_v2_{variant.lower()}: pretrained not found: {weight_path}')
        return model
    import pathlib; pathlib.PosixPath = pathlib.WindowsPath
    raw = torch.load(weight_path, map_location='cpu', weights_only=False)
    raw = raw.get('model', raw.get('state_dict', raw))

    remapped = _remap_keys_v2(raw)
    model_sd = model.state_dict()
    valid = {k: v for k, v in remapped.items()
             if k in model_sd and model_sd[k].shape == v.shape}
    missing, _ = model.load_state_dict(valid, strict=False)
    print(f'[OK] q_eformer_v2_{variant.lower()}: {len(valid)}/{len(model_sd)} loaded'
          f'  (missed={len(missing)}) from {weight_path}')
    return model


# ─── ファクトリ関数 ───────────────────────────────────────────────────────────────
def q_eformer_v2_s0(pretrained=False, **kwargs):
    model = Q_EfficientFormerV2(
        layers=EfficientFormer_depth['S0'], embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True, True], vit_num=2,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        e_ratios=expansion_ratios_S0, **kwargs)
    if pretrained: _load_pretrained(model, 'S0')
    return model


def q_eformer_v2_s1(pretrained=False, **kwargs):
    model = Q_EfficientFormerV2(
        layers=EfficientFormer_depth['S1'], embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True], vit_num=2,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        e_ratios=expansion_ratios_S1, **kwargs)
    if pretrained: _load_pretrained(model, 'S1')
    return model


def q_eformer_v2_s2(pretrained=False, **kwargs):
    model = Q_EfficientFormerV2(
        layers=EfficientFormer_depth['S2'], embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True], vit_num=4,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.02),
        e_ratios=expansion_ratios_S2, **kwargs)
    if pretrained: _load_pretrained(model, 'S2')
    return model


def q_eformer_v2_l(pretrained=False, **kwargs):
    model = Q_EfficientFormerV2(
        layers=EfficientFormer_depth['L'], embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True], vit_num=6,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.1),
        e_ratios=expansion_ratios_L, **kwargs)
    if pretrained: _load_pretrained(model, 'L')
    return model
