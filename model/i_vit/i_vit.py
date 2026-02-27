"""
i_vit.py
I-ViT の DeiT（Data-Efficient Image Transformer）量子化モデル定義
"""

import pretty_errors
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .quant_module import (QuantLinear, QuantAct, QuantConv2d, IntLayerNorm,
                                IntSoftmax, IntGELU, QuantMatMul, TerLinear)
    from .utils import load_weights_from_npz, trunc_normal_, to_2tuple
except ImportError:
    # python i_vit.py として直接実行する場合
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from model.i_vit.quant_module import (QuantLinear, QuantAct, QuantConv2d, IntLayerNorm,
                                           IntSoftmax, IntGELU, QuantMatMul, TerLinear)
    from model.i_vit.utils import load_weights_from_npz, trunc_normal_, to_2tuple


__all__ = ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224']


# ── DropPath（Stochastic Depth） ──────────────────────────────────────────────
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


# ── MLP層（量子化対応） ───────────────────────────────────────────────────────────────────
class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=IntGELU,
            drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TerLinear(in_features, hidden_features)
        self.act = act_layer()
        self.qact1 = QuantAct()
        self.fc2 = TerLinear(hidden_features, out_features)
        self.norm_mid = IntLayerNorm(hidden_features)
        self.qact_fc2 = QuantAct()
        self.qact2 = QuantAct(16)
        self.drop = nn.Dropout(drop)
        self.qact_gelu = QuantAct()

    def forward(self, x, act_scaling_factor):
        # 1. 線形層（TerLinear）[int8 -> int32]
        x, act_scaling_factor = self.fc1(x, act_scaling_factor)
        # 2. 量子化（8）[int32 -> int8]
        x, act_scaling_factor = self.qact_gelu(x, act_scaling_factor)
        # 3. 活性化関数（IntGELU）[int8 -> int8]
        x, act_scaling_factor = self.act(x, act_scaling_factor)
        # 4. 量子化（8）[int8 -> int8]
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        # # 4.5. 正規化（IntLayerNorm）+ 量子化（8）fc2(TerLinear)前のLN #! 事前学習での次の線形層に入力されるactの分布が異なるから、精度底まで改善しないかも
        # x, act_scaling_factor = self.norm_mid(x, act_scaling_factor)
        # x, act_scaling_factor = self.qact_fc2(x, act_scaling_factor)
        x = self.drop(x)
        # 5. 線形層（TerLinear）[int8 -> int32]
        x, act_scaling_factor = self.fc2(x, act_scaling_factor)
        # 6. 量子化（16）[int32 -> int16]
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        x = self.drop(x)
        return x, act_scaling_factor


# ── パッチ埋め込み層 ─────────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm_layer = norm_layer
        self.proj = QuantConv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.norm_layer:
            self.qact_before_norm = QuantAct()
            self.norm = norm_layer(embed_dim)
        self.qact = QuantAct(16)

    def forward(self, x, act_scaling_factor):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model "\
            f"({self.img_size[0]}*{self.img_size[1]})."
        # 1. 畳み込み（QuantConv2d）[int8 -> int32]
        x, act_scaling_factor = self.proj(x, act_scaling_factor)
        # 2. フラット化・転置[int32 -> int32]
        x = x.flatten(2).transpose(1, 2)
        if self.norm_layer:
            # 3. 量子化（8）[int32 -> int8]
            x, act_scaling_factor = self.qact_before_norm(x, act_scaling_factor)
            # 4. 正規化（IntLayerNorm）[int8 -> int32]
            x, act_scaling_factor = self.norm(x, act_scaling_factor)
        # 5. 量子化（16）[int32 -> int16]
        x, act_scaling_factor = self.qact(x, act_scaling_factor)
        return x, act_scaling_factor


# ── 自己アテンション層 ───────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = TerLinear(dim, dim * 3, bias=qkv_bias)
        self.qact1 = QuantAct()
        self.qact_attn1 = QuantAct()
        self.qact2 = QuantAct()
        self.proj = QuantLinear(dim, dim)
        self.qact3 = QuantAct(16)
        self.qact_softmax = QuantAct()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.int_softmax = IntSoftmax(16)

        self.matmul_1 = QuantMatMul()
        self.matmul_2 = QuantMatMul()

    def forward(self, x, act_scaling_factor):
        # 1. (Batch, Token, dim)を取得
        B, N, C = x.shape
        # 2. QKV生成（TerLinear）[int8 -> int32]
        x, act_scaling_factor = self.qkv(x, act_scaling_factor)
        # 3. 量子化（8）[int32 -> int8]
        x, act_scaling_factor_1 = self.qact1(x, act_scaling_factor)
        # 4. QKVを分割
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        # 5. アテンションマップの計算（QuantMatMul）[int8 × int8 -> int32]
        attn, act_scaling_factor = self.matmul_1(q, act_scaling_factor_1, k.transpose(-2, -1), act_scaling_factor_1)
        # 6. √dim で割る[int32 -> fp32]
        attn = attn * self.scale
        # 7. 量子化スケールも、√dimで割る
        act_scaling_factor = act_scaling_factor * self.scale
        # 8. アテンションマップの量子化（8）[fp32 -> int8]
        attn, act_scaling_factor = self.qact_attn1(attn, act_scaling_factor)
        # 9. ソフトマックス（IntSoftmax）[int8 -> int16]
        attn, act_scaling_factor = self.int_softmax(attn, act_scaling_factor)
        # 10. ドロップアウト[int16 -> int16]
        attn = self.attn_drop(attn)
        # 11. attn @ v の計算（QuantMatMul）[int16 × int8 -> int32]
        x, act_scaling_factor = self.matmul_2(attn, act_scaling_factor, v, act_scaling_factor_1)
        # 12. 次元変換[int32 -> int32]
        x = x.transpose(1, 2).reshape(B, N, C)
        # 13. 量子化（8）[int32 -> int8]
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        # 14. 線形層（QuantLinear）[int8 -> int32]
        x, act_scaling_factor = self.proj(x, act_scaling_factor)
        # 15. 量子化（16）[int32 -> int16]
        x, act_scaling_factor = self.qact3(x, act_scaling_factor)
        # 16. ドロップアウト
        x = self.proj_drop(x)
        return x, act_scaling_factor


# ── Transformerブロック ─────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qact1 = QuantAct()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QuantAct(16)
        self.norm2 = norm_layer(dim)
        self.qact3 = QuantAct()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.qact4 = QuantAct(16)

    def forward(self, x_1, act_scaling_factor_1):
        # 1. 正規化（IntLayerNorm）[int16 -> int32]
        x, act_scaling_factor = self.norm1(x_1, act_scaling_factor_1)
        # 2. 量子化（8）[int32 -> int8]
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        # 3. Attentionブロック（Attention）[int8 -> int16]
        x, act_scaling_factor = self.attn(x, act_scaling_factor)
        # 4. ドロップアウト[int16 -> int16]
        x = self.drop_path(x)
        # 5. 量子化（16）[int16 + int16 -> int16]
        x_2, act_scaling_factor_2 = self.qact2(x, act_scaling_factor, x_1, act_scaling_factor_1)
        # 6. 正規化（IntLayerNorm）[int16 -> int32]
        x, act_scaling_factor = self.norm2(x_2, act_scaling_factor_2)
        # 7. 量子化（8）[int32 -> int8]
        x, act_scaling_factor = self.qact3(x, act_scaling_factor)
        # 8. MLP層（Mlp）[int8 -> int16]
        x, act_scaling_factor = self.mlp(x, act_scaling_factor)
        # 9. ドロップアウト[int16 -> int16]
        x = self.drop_path(x)
        # 10. 量子化（16）[int16 + int16 -> int16]
        x, act_scaling_factor = self.qact4(x, act_scaling_factor, x_2, act_scaling_factor_2)
        return x, act_scaling_factor


# ── Vision Transformerモデル ───────────────────────────────────────────────────────────
class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.qact_input = QuantAct()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.qact_pos = QuantAct(16)
        self.qact1 = QuantAct(16)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=IntGELU,
                    norm_layer=norm_layer
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.qact2 = QuantAct()

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            QuantLinear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        # 1. 入力xを量子化（8）[fp32 -> int8]
        x, act_scaling_factor = self.qact_input(x)
        # 2. パッチ埋め込み（PatchEmbed）[int8 -> int16]
        x, act_scaling_factor = self.patch_embed(x, act_scaling_factor)
        # 3. クラストークン
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 4. 位置埋め込み量子化（16）
        x_pos, act_scaling_factor_pos = self.qact_pos(self.pos_embed)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor, x_pos, act_scaling_factor_pos)
        # 5. ドロップアウト[int16 -> int16]
        x = self.pos_drop(x)
        # 6. エンコーダーブロック（Block）[int16 -> int16]
        for blk in self.blocks:
            x, act_scaling_factor = blk(x, act_scaling_factor)
        # 7. 正規化（IntLayerNorm）[int16 -> int32]
        x, act_scaling_factor = self.norm(x, act_scaling_factor)
        # 8. クラストークン取り出し
        x = x[:, 0]
        # 9. クラストークン量子化（8）[int32 -> int8]
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        # 10. pre_logits（DeiT_tiny_patch16_224 だと等価関数）
        x = self.pre_logits(x)
        return x, act_scaling_factor

    def forward(self, x):
        x, act_scaling_factor = self.forward_features(x)
        x, act_scaling_factor = self.head(x, act_scaling_factor)
        return x


# ── DeiT-Tiny モデル構築 ───────────────────────────────────────────────────────────────
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        state_dict = checkpoint["model"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        model.load_state_dict(state_dict, strict=False)
    return model


# ── DeiT-Small モデル構築 ──────────────────────────────────────────────────────────────
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


# ── DeiT-Base モデル構築 ───────────────────────────────────────────────────────────────
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


# ── torchinfo  ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torchinfo import summary

    MODELS = {
        "deit_tiny_patch16_224":  deit_tiny_patch16_224,
        "deit_small_patch16_224": deit_small_patch16_224,
        "deit_base_patch16_224":  deit_base_patch16_224,
    }
    IMG_SIZE    = 224
    IN_CHANS    = 3
    NUM_CLASSES = 1000
    BATCH_SIZE  = 1

    for name, fn in MODELS.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        model = fn(pretrained=False, num_classes=NUM_CLASSES)
        model.eval()
        summary(
            model,
            input_size=(BATCH_SIZE, IN_CHANS, IMG_SIZE, IMG_SIZE),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=3,
            verbose=1,
        )

