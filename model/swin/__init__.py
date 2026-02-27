"""
model/swin/__init__.py
Swin Transformer 量子化モデルのエクスポート
"""

import pretty_errors

from .swin import (
    SwinTransformer,
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
)
from .quant_module import QuantAct  # freeze/unfreeze 判定用

__all__ = [
    "SwinTransformer",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
]

# ── モデル名 → 構築関数マッピング ─────────────────────────────────────────────────────
MODEL_MAP = {
    "swin_tiny":  swin_tiny_patch4_window7_224,
    "swin_small": swin_small_patch4_window7_224,
    "swin_base":  swin_base_patch4_window7_224,
}


def get_model(name: str):
    """モデル名から構築関数を返す。存在しない場合は KeyError を送出する。"""
    if name not in MODEL_MAP:
        raise KeyError(f"Unknown swin model: '{name}'. Available: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[name]
