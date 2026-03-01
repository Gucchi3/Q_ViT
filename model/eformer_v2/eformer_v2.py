"""
eformer_v2.py
EfficientFormer V2 モデル定義（量子化なし）

論文: Rethinking Vision Transformers for MobileNet Size and Speed
重み: ./data/eformer_{s0,s1,s2,l}_450.pth
"""

import os
import copy
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import to_2tuple, trunc_normal_
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from model.eformer_v2.utils import to_2tuple, trunc_normal_

__all__ = [
    'EfficientFormerV2',
    'eformer_v2_s0', 'eformer_v2_s1', 'eformer_v2_s2', 'eformer_v2_l',
]

# ── モデルサイズ設定 ────────────────────────────────────────────────────────────────
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

expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}

# ── 事前学習済み重みファイルパス ────────────────────────────────────────────────────
PRETRAINED_WEIGHTS = {
    'S0': './data/eformer_s0_450.pth',
    'S1': './data/eformer_s1_450.pth',
    'S2': './data/eformer_s2_450.pth',
    'L':  './data/eformer_l_450.pth',
}


# ── DropPath（Stochastic Depth） ──────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ── ステム ───────────────────────────────────────────────────────────────────────────
def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )


# ── Attention4D（全ステージ共通の 2D アテンション） ──────────────────────────────────
class Attention4D(nn.Module):
    """4D（B,C,H,W）テンソルのまま処理するアテンション（V2 の特徴）"""
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4, resolution=7, act_layer=nn.ReLU, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None
        self.N = self.resolution ** 2
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, 1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, 1)
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim),
        )
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        attn = ((q @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab))
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)
        x = (attn @ v)
        out = x.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution) + v_local
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.proj(out)
        return out


# ── LGQuery（Embedding 用ローカル+グローバルクエリ） ─────────────────────────────────
class LGQuery(nn.Module):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.proj(self.local(x) + self.pool(x))


# ── Attention4DDownsample（Embedding 用ダウンサンプリングアテンション） ──────────────
class Attention4DDownsample(nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4, resolution=7, out_dim=None, act_layer=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.resolution = resolution
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.out_dim = out_dim if out_dim is not None else dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim),
        )
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2),
                )
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        attn = ((q @ k) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training else self.ab
        ))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, self.resolution2, self.resolution2) + v_local
        out = self.proj(out)
        return out


# ── Embedding（ダウンサンプリング/軽量/アテンション付きの 3 種） ─────────────────────
class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1,
                in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                light=False, asub=False, resolution=None,
                act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub
        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, 3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, 1, stride=2),
                nn.BatchNorm2d(embed_dim),
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim, resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            return self.new_proj(x) + self.skip(x)
        elif self.asub:
            return self.attn(x) + self.bn(self.conv(x))
        else:
            return self.norm(self.proj(x))


# ── Mlp（Conv ベース、mid_conv オプション付き） ───────────────────────────────────────
class Mlp(nn.Module):
    """1×1 Conv を用いた MLP（mid_conv=True でデプスワイズ Conv を中間に挿入）"""
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features,
                                  kernel_size=3, stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


# ── AttnFFN ブロック（アテンション付き FFN ブロック） ─────────────────────────────────
class AttnFFN(nn.Module):
    """Attention4D + MLP（attn ステージ用）"""
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution,
                                       act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


# ── FFN ブロック（MLP のみ、アテンションなし） ─────────────────────────────────────────
class FFN(nn.Module):
    """MLP のみ（CNN ステージ FFN）"""
    def __init__(self, dim, pool_size=3, mlp_ratio=4., act_layer=nn.GELU,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


# ── ステージ構築ヘルパー ───────────────────────────────────────────────────────────
def eformer_block(dim, index, layers, pool_size=3, mlp_ratio=4.,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                  drop_rate=.0, drop_path_rate=0.,
                  use_layer_scale=True, layer_scale_init_value=1e-5,
                  vit_num=1, resolution=7, e_ratios=None):
    """V2 の各ステージを構成するブロックの Sequential を構築する"""
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio_blk = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            # AttnFFN ブロック
            stride = 2 if index == 2 else None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio_blk,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution, stride=stride,
            ))
        else:
            # FFN ブロック
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio_blk,
                act_layer=act_layer, drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    return nn.Sequential(*blocks)


# ── EfficientFormer V2 モデル本体 ──────────────────────────────────────────────────
class EfficientFormerV2(nn.Module):
    """
    EfficientFormer V2 本体（全ステージ 4D）。

    Args:
        layers      : 各ステージのブロック数リスト（長さ 4）
        embed_dims  : 各ステージの埋め込み次元リスト（長さ 4）
        vit_num     : アテンション付ブロック（AttnFFN）の数
        distillation: 蒸留ヘッドを追加するか
        resolution  : 入力画像解像度（通常 224）
        e_ratios    : ステージ毎の MLP 拡張比辞書
        in_chans    : 入力チャネル数
        drop_rate   : ドロップアウト率
        drop_path_rate: ストキャスティック深度率
        num_classes : 分類クラス数
    """

    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None,
                 pool_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 vit_num=0, distillation=False, resolution=224,
                 e_ratios=expansion_ratios_L, in_chans=3, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = stem(in_chans, embed_dims[0], act_layer=nn.ReLU)

        network = []
        for i in range(len(layers)):
            stage = eformer_block(
                embed_dims[i], i, layers,
                pool_size=pool_size, mlp_ratio=mlp_ratios,
                act_layer=act_layer, norm_layer=norm_layer,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=math.ceil(resolution / (2 ** (i + 2))),
                vit_num=vit_num, e_ratios=e_ratios,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                asub = (i >= 2)
                network.append(Embedding(
                    patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                    resolution=math.ceil(resolution / (2 ** (i + 2))),
                    asub=asub, act_layer=nn.ReLU, norm_layer=norm_layer,
                ))
        self.network = nn.ModuleList(network)

        # 最終正規化（BN）と分類ヘッド
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.dist = distillation
        if self.dist:
            self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x):
        # 1. ステム（Conv × 2）で空間解像度を 1/4 に縮小
        x = self.patch_embed(x)
        # 2. ネットワークブロック（FFN / AttnFFN を順次実行、全 4D テンソル）
        x = self.forward_tokens(x)
        # 3. 最終正規化（BatchNorm2d、x は [B, C, H, W] の 4D テンソル）
        x = self.norm(x)
        # 4. グローバル平均プーリング（空間次元を平均化）-> [B, C]
        x_cls = x.flatten(2).mean(-1)
        # 5. 分類ヘッド
        if self.dist:
            cls_out = self.head(x_cls), self.dist_head(x_cls)
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x_cls)
        return cls_out


# ── pretrained 重みローダー ────────────────────────────────────────────────────────
def _load_pretrained(model: EfficientFormerV2, variant: str):
    weight_path = PRETRAINED_WEIGHTS.get(variant, '')
    if not weight_path or not os.path.exists(weight_path):
        print(f"[Warning] eformer_v2_{variant.lower()}: pretrained weight not found: {weight_path}")
        return model
    state = torch.load(weight_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # num_classes 違いによるサイズ不一致を防ぐため head を除去
    state = {k: v for k, v in state.items() if not k.startswith('head.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Info] eformer_v2_{variant.lower()}: missing keys (probably head): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    print(f"[OK] eformer_v2_{variant.lower()}: loaded pretrained weights from {weight_path}")
    return model


# ── 公開ファクトリ関数 ─────────────────────────────────────────────────────────────
def eformer_v2_s0(pretrained=False, **kwargs):
    """EfficientFormerV2-S0（3.5M パラメータ相当）"""
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S0'],
        embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True, True],
        vit_num=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        e_ratios=expansion_ratios_S0,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'S0')
    return model


def eformer_v2_s1(pretrained=False, **kwargs):
    """EfficientFormerV2-S1（6.1M パラメータ相当）"""
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S1'],
        embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True],
        vit_num=2, drop_path_rate=kwargs.pop('drop_path_rate', 0.0),
        e_ratios=expansion_ratios_S1,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'S1')
    return model


def eformer_v2_s2(pretrained=False, **kwargs):
    """EfficientFormerV2-S2（12M パラメータ相当）"""
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S2'],
        embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True],
        vit_num=4, drop_path_rate=kwargs.pop('drop_path_rate', 0.02),
        e_ratios=expansion_ratios_S2,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'S2')
    return model


def eformer_v2_l(pretrained=False, **kwargs):
    """EfficientFormerV2-L（26M パラメータ相当）"""
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['L'],
        embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True],
        vit_num=6, drop_path_rate=kwargs.pop('drop_path_rate', 0.1),
        e_ratios=expansion_ratios_L,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'L')
    return model
