# DeiT モデル構成

本プロジェクトでは **I-ViT** の量子化 DeiT を `model/i_vit/i_vit.py` に実装しています。  
モデル構成を変更したい場合は、同ファイル内の各ビルダー関数を直接編集してください。

---

## モデル構成表

| モデル名 | patch_size | embed_dim | depth | num_heads | mlp_ratio | img_size | パラメータ数（概算） |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeiT-Tiny  | 16 | 192  | 12 | 3  | 4.0 | 224 | ≈ 5.7 M |
| DeiT-Small | 16 | 384  | 12 | 6  | 4.0 | 224 | ≈ 22 M  |
| DeiT-Base  | 16 | 768  | 12 | 12 | 4.0 | 224 | ≈ 86 M  |

> **注意**: `num_heads = embed_dim / 64` となっており、head ごとの次元数は 64 に固定されています。

---

## データフロー

```
Image (H×W×3)
  │
  ▼
PatchEmbed  (patch_size=16 で 14×14=196 トークン)
  │
  ▼
+ Positional Embedding  (CLS トークン含む 197×embed_dim)
  │
  ▼
TransformerBlock × depth
  ├─ IntLayerNorm
  ├─ QKV (QuantLinear) + IntSoftmax + QuantMatMul
  └─ IntLayerNorm → Mlp (QuantLinear × 2) + IntGELU
  │
  ▼
CLS トークン → Head (Linear) → クラス logits
```

---

## 量子化パラメータ（デフォルト）

| 設定項目 | キー（config.json） | デフォルト値 |
|:---|:---|:---:|
| 入力活性化量子化ビット幅 | `quant_bit` (QuantAct) | 8 bit |
| 重み量子化ビット幅 | `weight_bit` (QuantLinear) | 8 bit |
| 行列積量子化ビット幅 | `quant_bit` (QuantMatMul) | 8 bit |
| IntSoftmax 最大整数ビット幅 | `i_softmax_quant_bit` | 22 bit |
| IntLayerNorm フラグ | — | True（固定） |

---

## ビルダー関数（i_vit.py）

```python
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    ...

def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    ...

def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    ...
```

config.json の `"MODEL"` に `"deit_tiny"` / `"deit_small"` / `"deit_base"` のいずれかを指定してください。
