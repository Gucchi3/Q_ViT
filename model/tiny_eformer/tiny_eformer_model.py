import math
import itertools

import pretty_errors
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import to_2tuple, trunc_normal_
except ImportError:
    from utils import to_2tuple, trunc_normal_


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

class Stem(nn.Module):
    """
    stem: Conv3x3 BN ReLU x2 (stride=2 ずつ)
    入力 (B, in_chs, H, W) → 出力 (B, out_chs, H/4, W/4)
    """
    def __init__(self, in_chs: int, out_chs: int, act_layer=nn.ReLU6):
        super().__init__()
        mid = out_chs // 2
        # conv1: 通常 Conv
        self.conv1   = nn.Conv2d(in_chs, mid, kernel_size=3, stride=2, padding=1)
        self.bn1     = nn.BatchNorm2d(mid)
        self.act1    = act_layer()
        # conv2: DW(stride=2) + PW
        self.conv2_dw   = nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1, groups=mid)
        self.bn2_dw     = nn.BatchNorm2d(mid)
        self.act2_dw    = act_layer()
        self.conv2_pw   = nn.Conv2d(mid, out_chs, kernel_size=1)
        self.bn2        = nn.BatchNorm2d(out_chs)
        self.act2       = act_layer()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2_pw(self.act2_dw(self.bn2_dw(self.conv2_dw(x))))))
        return x


# ─────────────────────────────────────────────────────────────────────────────
class Mlp(nn.Module):
    """
    MLP: fc1+BN+act → [mid(DW)+BN+act] → fc2+BN
    mid_conv=True のとき DW-Conv 3x3 の中間層が入る（AttnFFN/FFN で使用）
    """
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=nn.ReLU6,
                 drop: float = 0., mid_conv: bool = False):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv

        self.fc1   = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act   = act_layer()
        self.drop  = nn.Dropout(drop)

        if mid_conv:
            self.mid      = nn.Conv2d(hidden_features, hidden_features, 3, stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.fc2   = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = nn.BatchNorm2d(out_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.norm1(self.fc1(x)))
        if self.mid_conv:
            x = self.act(self.mid_norm(self.mid(x)))
        x = self.drop(x)
        x = self.norm2(self.fc2(x))
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
class LGQuery(nn.Module):
    """
    Local-Global Query (Attention4DDownsample 専用)
    DW-Conv(stride=2) + AvgPool(stride=2) → 加算 → proj(Conv1x1+BN)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.pool  = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim)
        self.proj  = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.proj(self.local(x) + self.pool(x))


# ─────────────────────────────────────────────────────────────────────────────
class Attention4D(nn.Module):
    """
    AttnFFN ブロック内の 4D Self-Attention
    stride 設定時: x 全体を stride_conv で縮小後に Q/K/V を生成し、最後に upsample
    talking_head1/2 あり（V2 の特徴）
    """
    def __init__(self, dim: int = 384, key_dim: int = 96, num_heads: int = 1,
                attn_ratio: int = 3, resolution: int = 7,
                act_layer=nn.ReLU, stride: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = key_dim ** -0.5
        self.key_dim   = key_dim
        self.d         = int(attn_ratio * key_dim)
        self.dh        = self.d * num_heads
        self.nh_kd     = key_dim * num_heads

        if stride is not None:
            self.resolution  = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim))
            self.upsample    = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution  = resolution
            self.stride_conv = None
            self.upsample    = None

        self.N = self.resolution ** 2

        self.q = nn.Sequential(
            nn.Conv2d(dim, self.nh_kd, 1),
            nn.BatchNorm2d(self.nh_kd),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.nh_kd, 1),
            nn.BatchNorm2d(self.nh_kd),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1),
            nn.BatchNorm2d(self.dh),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=1, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )

        self.talking_head1 = nn.Conv2d(num_heads, num_heads, 1)
        self.talking_head2 = nn.Conv2d(num_heads, num_heads, 1)

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim)
            )

        pts = list(itertools.product(range(self.resolution), range(self.resolution)))
        attention_offsets = {}
        idxs = []
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
        if mode and hasattr(self, 'ab'):
            del self.ab
        elif not mode:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape

        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)

        q = F.normalize(q, p=1, dim=-1)
        k = F.normalize(k, p=1, dim=-2)
        
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        # ab = (self.ab if hasattr(self, 'ab') else self.attention_biases[:, self.attention_bias_idxs])
        # attn = (q @ k) * self.scale + ab.to(q.device)
        attn = (q @ k) * self.scale
        # attn = self.talking_head1(attn)
        # attn = attn.softmax(dim=-1)
        # attn = self.talking_head2(attn)

        out = (attn @ v).transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution)
        out = out + v_local

        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class Attention4DDownsample(nn.Module):
    """
    Embedding(asub=True) 専用のダウンサンプリング付き 4D Attention
    Q は LGQuery (stride=2)、K/V は元解像度の x から生成
    talking_head なし（原本通り）
    """
    def __init__(self, dim: int = 384, key_dim: int = 96, num_heads: int = 1,
                 attn_ratio: int = 3, resolution: int = 7,
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

        self.q = LGQuery(dim, self.nh_kd)

        self.k = nn.Sequential(
            nn.Conv2d(dim, self.nh_kd, 1),
            nn.BatchNorm2d(self.nh_kd),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1),
            nn.BatchNorm2d(self.dh),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=2, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim),
        )

        pts_q = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        pts_k = list(itertools.product(range(resolution), range(resolution)))
        attention_offsets = {}
        idxs = []
        for p1 in pts_q:
            for p2 in pts_k:
                size = 1
                off = (
                    abs(p1[0] * math.ceil(resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(resolution / self.resolution2) - p2[1] + (size - 1) / 2),
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

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)

        q = F.normalize(q, p=1, dim=-1)
        k = F.normalize(k, p=1, dim=-2)

        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        # ab = (self.ab if hasattr(self, 'ab') else self.attention_biases[:, self.attention_bias_idxs])
        # attn = (q @ k) * self.scale + ab.to(q.device)
        attn = (q @ k) * self.scale
        # attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(2, 3).reshape(B, self.dh, self.resolution2, self.resolution2)
        out = out + v_local
        out = self.proj(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class Embedding(nn.Module):
    """
    Embedding (ステージ間ダウンサンプリング)
      asub=False: Conv+BN (norm)
      asub=True : Attention4DDownsample + Conv+BN を並列加算
    """
    def __init__(self, in_chs: int, out_chs: int,
                 patch_size: int = 3, stride: int = 2, padding: int = 1,
                 asub: bool = False, resolution: int = None,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.asub = asub

        if asub:
            self.attn = Attention4DDownsample(
                dim=in_chs, out_dim=out_chs,
                resolution=resolution, act_layer=act_layer)
            ps = to_2tuple(patch_size)
            st = to_2tuple(stride)
            pd = to_2tuple(padding)
            # DS-Conv: DW + PW
            self.conv_dw     = nn.Conv2d(in_chs, in_chs, kernel_size=ps, stride=st, padding=pd, groups=in_chs)
            self.conv_bn_dw  = nn.BatchNorm2d(in_chs)
            self.conv_act_dw = act_layer()
            self.conv_pw     = nn.Conv2d(in_chs, out_chs, kernel_size=1)
            self.bn          = norm_layer(out_chs) if norm_layer else nn.Identity()
        else:
            ps = to_2tuple(patch_size)
            st = to_2tuple(stride)
            pd = to_2tuple(padding)
            # DS-Conv: DW + PW
            self.proj_dw     = nn.Conv2d(in_chs, in_chs, kernel_size=ps, stride=st, padding=pd, groups=in_chs)
            self.proj_bn_dw  = nn.BatchNorm2d(in_chs)
            self.proj_act_dw = act_layer()
            self.proj_pw     = nn.Conv2d(in_chs, out_chs, kernel_size=1)
            self.norm        = norm_layer(out_chs) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.asub:
            return self.attn(x) + self.bn(self.conv_pw(self.conv_act_dw(self.conv_bn_dw(self.conv_dw(x)))))
        else:
            return self.norm(self.proj_pw(self.proj_act_dw(self.proj_bn_dw(self.proj_dw(x)))))


# ─────────────────────────────────────────────────────────────────────────────
class FFN(nn.Module):
    """
    FFN ブロック: x + drop_path(layer_scale * mlp(x))
    mid_conv=True の Mlp を使用
    """
    def __init__(self, dim: int, mlp_ratio: float = 3., act_layer=nn.ReLU6,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init_value: float = 1e-5):
        super().__init__()
        
        self.mlp       = Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
class AttnFFN(nn.Module):
    """
    AttnFFN ブロック: token_mixer(Attention4D) + mlp(Mlp)
    x → attn残差(layer_scale_1) → mlp残差(layer_scale_2)
    """
    def __init__(self, dim: int, mlp_ratio: float = 3., act_layer=nn.ReLU6,
                 drop: float = 0., drop_path: float = 0.,
                 use_layer_scale: bool = True, layer_scale_init_value: float = 1e-5,
                 resolution: int = 7, stride: int = None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# ── TinyEFormer ──
# ══════════════════════════════════════════════════════════════════════════════

class TinyEFormer(nn.Module):
    """
    Tiny EfficientFormer モデル（通常版・量子化なし）。
    __init__ に self.xxx = <モジュール> を1つずつ記述し、
    forward で x を渡しながら接続する。

    利用できるモジュール:
        Stem, FFN, AttnFFN, Embedding,
        Attention4D, Attention4DDownsample,
        LGQuery, Mlp, DropPath,
        nn.BatchNorm2d, nn.Linear 等 (通常の PyTorch モジュール)
    """

    def __init__(self, num_classes: int = 10, img_size: int = 224, **kwargs):
        super().__init__()
        stage3_resolution = math.ceil(img_size / 16)
        stage4_resolution = math.ceil(img_size / 32)

        # 1. Stem: 3 → 16, img_size → ceil(img_size / 4)
        self.stem = Stem(in_chs=3, out_chs=16)
        # 2. FFN: dim=16
        self.ffn1 = FFN(dim=16, mlp_ratio=3)
        # 3. Embedding (asub=False): 16 → 32
        self.emb1 = Embedding(in_chs=16, out_chs=32, asub=False)
        # 4. FFN: dim=32
        self.ffn2 = FFN(dim=32)
        # 5. Embedding (asub=False): 32 → 48
        self.emb2 = Embedding(in_chs=32, out_chs=48, asub=False)
        # 6. AttnFFN (stride=2, 内部ダウンサンプル後アップサンプル)
        self.attn4d_s = AttnFFN(dim=48, resolution=stage3_resolution, stride=2)
        # self.attn4d_s = FFN(dim=48)
        # 7. Embedding (asub=True): 48 → 64
        self.attn4d_ds = Embedding(
            in_chs=48, out_chs=64, asub=True, resolution=stage3_resolution)
        # self.attn4d_ds = Embedding(in_chs=48, out_chs=64, asub=False)
        # 8. AttnFFN: dim=64, res=ceil(img_size / 32)
        self.attn4d = AttnFFN(dim=64, resolution=stage4_resolution)
        # Head
        self.norm = nn.BatchNorm2d(64)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.ffn1(x)
        x = self.emb1(x)
        x = self.ffn2(x)
        x = self.emb2(x)
        x = self.attn4d_s(x)
        x = self.attn4d_ds(x)
        x = self.attn4d(x)
        x = self.norm(x)
        x = x.mean(dim=[2, 3])   # GlobalAvgPool → (B, 64)
        x = self.head(x)         # (B, num_classes)
        return x


# ── ファクトリ関数 ────────────────────────────────────────────────────────────

def tiny_eformer(pretrained: bool = False, num_classes: int = 10, **kwargs):
    """config の MODEL="tiny_eformer" で呼ばれるエントリポイント"""
    model = TinyEFormer(num_classes=num_classes, **kwargs)
    return model


# ── 動作確認 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchinfo import summary

    input_size = 160
    model = TinyEFormer(num_classes=10, img_size=input_size)
    summary(model, input_size=(1, 3, input_size, input_size))
