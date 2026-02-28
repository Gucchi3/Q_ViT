"""
model/i_vit/__init__.py
DeiT 量子化モデルのエクスポート
"""

import pretty_errors

from .i_vit import (
    VisionTransformer,
    deit_tiny_patch16_224,
    deit_small_patch16_224,
    deit_base_patch16_224,
    deit_cus,
)
from .quant_module import QuantAct  # freeze/unfreeze 判定用

__all__ = [
    "VisionTransformer",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit_cus",
]

# ── モデル名 → 構築関数マッピング ─────────────────────────────────────────────────────
MODEL_MAP = {
    "deit_tiny":  deit_tiny_patch16_224,
    "deit_small": deit_small_patch16_224,
    "deit_base":  deit_base_patch16_224,
    "deit_cus":   deit_cus,
}


def get_model(name: str):
    """モデル名から構築関数を返す。存在しない場合は KeyError を送出する。"""
    if name not in MODEL_MAP:
        raise KeyError(f"Unknown i_vit model: '{name}'. Available: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[name]
