import os
import copy
import itertools

import torch
import torch.nn as nn

try:
    from .utils import to_2tuple, trunc_normal_
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from model.eformer_v1.utils import to_2tuple, trunc_normal_

__all__ = [
    'EfficientFormer',
    'eformer_v1_l1', 'eformer_v1_l3', 'eformer_v1_l7',
]

# ── モデルサイズ設定 ────────────────────────────────────────────────────────────────
EfficientFormer_width = {
    'l1': [48,  96,  224, 448],
    'l3': [64,  128, 320, 512],
    'l7': [96,  192, 384, 768],
}

EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}

# ── 事前学習済み重みファイルパス ────────────────────────────────────────────────────
PRETRAINED_WEIGHTS = {
    'l1': './data/efficientformer_l1_1000d.pth',
    'l3': './data/efficientformer_l3_300d.pth',
    'l7': './data/efficientformer_l7_300d.pth',
}


# ── DropPath（Stochastic Depth） ──────────────────────────────────────────────────
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
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


# ── アテンション層（Meta3D 用・トークンベース） ────────────────────────────────────────
class Attention(nn.Module):
    """EfficientFormer V1 のトークンアテンション（解像度依存の位置バイアス付き）"""
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4, resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.N = resolution ** 2
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        # 解像度に依存した位置バイアステーブルの構築
        points = list(itertools.product(range(resolution), range(resolution)))
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
        self.register_buffer('attention_bias_idxs',torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


# ── ステム（最初の2段 Conv+BN+ReLU） ────────────────────────────────────────────────
def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(),
    )


# ── ダウンサンプリング埋め込み（ステージ間） ─────────────────────────────────────────
class Embedding(nn.Module):
    """ステージ間のダウンサンプリング（Conv2d + BN）"""
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


# ── フラット化（4D -> 3D, Meta3D 前処理） ──────────────────────────────────────────
class Flat(nn.Module):
    def forward(self, x):
        # (B, C, H, W) -> (B, H*W, C)
        return x.flatten(2).transpose(1, 2)


# ── プーリングトークンミキサー（Meta4D 用） ──────────────────────────────────────────
class Pooling(nn.Module):
    """AvgPool2d - x によるローカル集約（プーリングトークンミキサー）"""
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


# ── Conv ベース MLP（Meta4D 用） ─────────────────────────────────────────────────────
class Mlp(nn.Module):
    """1×1 Conv2d を用いた MLP（Meta4D ステージ用）"""
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


# ── Linear MLP（Meta3D 用） ──────────────────────────────────────────────────────────
class LinearMlp(nn.Module):
    """nn.Linear を用いた MLP（Meta3D トランスフォーマーステージ用）"""
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ── Meta3D ブロック（トランスフォーマーステージ） ────────────────────────────────────
class Meta3D(nn.Module):
    """トークン次元でのアテンションブロック（最終ステージの ViT 部分）"""
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Meta4D ブロック（CNN ステージ） ──────────────────────────────────────────────────
class Meta4D(nn.Module):
    """プーリングトークンミキサー + Conv MLP のブロック（CNN ステージ用）"""
    def __init__(self, dim, pool_size=3, mlp_ratio=4., act_layer=nn.GELU,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


# ── ステージ構築ヘルパー ───────────────────────────────────────────────────────────
def meta_blocks(dim, index, layers, pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    """ステージを構成するブロックの Sequential を構築する"""
    blocks = []
    # stage3 の最初に vit_num == layers[3] のとき先頭に Flat を挿入
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            # Meta3D ブロック（トランスフォーマー）
            blocks.append(Meta3D(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
        else:
            # Meta4D ブロック（CNN）
            blocks.append(Meta4D(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            # Meta4D の最後に Flat を挿入（Meta3D への遷移直前）
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())
    return nn.Sequential(*blocks)


# ── EfficientFormer V1 モデル本体 ──────────────────────────────────────────────────
class EfficientFormer(nn.Module):
    """
    EfficientFormer V1 本体。

    Args:
        layers      : 各ステージのブロック数リスト（長さ 4）
        embed_dims  : 各ステージの埋め込み次元リスト（長さ 4）
        mlp_ratios  : MLP 拡張比（int, ステージ共通）
        downsamples : ステージ間にダウンサンプリングを挿入するかのリスト
        pool_size   : プーリングカーネルサイズ
        norm_layer  : Meta3D 用正規化レイヤー
        act_layer   : 活性化関数
        num_classes : 分類クラス数
        vit_num     : stage3 の末尾に置く Meta3D ブロック数
        drop_rate   : ドロップアウト率
        drop_path_rate: ストキャスティック深度率
        use_layer_scale      : レイヤースケール機能
        layer_scale_init_value: レイヤースケール初期値
        in_chans    : 入力チャネル数（通常 3）
        distillation: 蒸留ヘッドを追加するか
    """

    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None,
                 pool_size=3, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=1000, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 vit_num=0, distillation=False, in_chans=3, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # ステム：2 段 Conv (stride 4 合計)
        self.patch_embed = stem(in_chans, embed_dims[0])

        # ネットワーク構築
        network = []
        for i in range(len(layers)):
            stage = meta_blocks(
                embed_dims[i], i, layers,
                pool_size=pool_size, mlp_ratio=mlp_ratios,
                act_layer=act_layer, norm_layer=norm_layer,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                vit_num=vit_num,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(Embedding(
                    patch_size=down_patch_size, stride=down_stride,
                    padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                ))
        self.network = nn.ModuleList(network)

        # 最終正規化（LayerNorm）と分類ヘッド
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # 蒸留ヘッド（distillation=True のとき）
        self.dist = distillation
        if self.dist:
            self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x):
        # 1. ステム（Conv × 2）で空間解像度を 1/4 に縮小
        x = self.patch_embed(x)
        # 2. ネットワークブロック（Meta4D / Flat / Meta3D の順で実行）
        x = self.forward_tokens(x)
        # 3. 最終正規化（LayerNorm、x は [B, N, C] の 3D テンソル）
        x = self.norm(x)
        # 4. グローバル平均プーリング（トークン次元 N を平均化）
        x_cls = x.mean(dim=1)
        # 5. 分類ヘッド
        if self.dist:
            cls_out = self.head(x_cls), self.dist_head(x_cls)
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x_cls)
        return cls_out


# ── pretrained 重みローダー ────────────────────────────────────────────────────────
def _load_pretrained(model: EfficientFormer, variant: str):
    """./data/ 以下の .pth ファイルから事前学習済み重みを読み込む。
    ImageNet (1000 クラス) 用の重みなので、head は strict=False で読み飛ばす。
    """
    weight_path = PRETRAINED_WEIGHTS.get(variant, '')
    if not weight_path or not os.path.exists(weight_path):
        print(f"[Warning] eformer_v1_{variant}: pretrained weight not found: {weight_path}")
        return model
    state = torch.load(weight_path, map_location='cpu', weights_only=False)
    # 保存形式に応じてモデルキーを取り出す
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # num_classes 違いによるサイズ不一致を防ぐため head / dist_head を除去
    state = {k: v for k, v in state.items()
             if not k.startswith('head.') and not k.startswith('dist_head.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Info] eformer_v1_{variant}: missing keys (probably head / dist_head): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    print(f"[OK] eformer_v1_{variant}: loaded pretrained weights from {weight_path}")
    return model


# ── 公開ファクトリ関数 ─────────────────────────────────────────────────────────────
def eformer_v1_l1(pretrained=False, **kwargs):
    """EfficientFormer-L1（12M パラメータ相当）"""
    model = EfficientFormer(
        layers=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'l1')
    return model


def eformer_v1_l3(pretrained=False, **kwargs):
    """EfficientFormer-L3（31M パラメータ相当）"""
    model = EfficientFormer(
        layers=EfficientFormer_depth['l3'],
        embed_dims=EfficientFormer_width['l3'],
        downsamples=[True, True, True, True],
        vit_num=4,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'l3')
    return model


def eformer_v1_l7(pretrained=False, **kwargs):
    """EfficientFormer-L7（82M パラメータ相当）"""
    model = EfficientFormer(
        layers=EfficientFormer_depth['l7'],
        embed_dims=EfficientFormer_width['l7'],
        downsamples=[True, True, True, True],
        vit_num=8,
        **kwargs,
    )
    if pretrained:
        _load_pretrained(model, 'l7')
    return model
