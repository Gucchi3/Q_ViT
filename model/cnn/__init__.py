"""
model/cnn/__init__.py
診断用 CNN モデルのエクスポート
"""

from .cnn import TinyCNN, tiny_cnn

__all__ = ["TinyCNN", "tiny_cnn"]

MODEL_MAP = {
    "tiny_cnn": tiny_cnn,
}


def get_model(name: str):
    if name not in MODEL_MAP:
        raise KeyError(f"Unknown cnn model: '{name}'. Available: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[name]
