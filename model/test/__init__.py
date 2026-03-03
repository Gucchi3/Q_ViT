from .test import TestCNN, test_cnn
from .module import (
    QuantAct,
    QuantLinear,
    QuantBNConv2d,
    QuantConv2d,
)

MODEL_MAP = {
    "test":    test_cnn,
    "testcnn": test_cnn,
}


def get_model(name: str):
    """utils.build_model から呼ばれる。name に対応するファクトリ関数を返す。"""
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"test: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]


__all__ = [
    "TestCNN",
    "test_cnn",
    "get_model",
    "QuantAct",
    "QuantLinear",
    "QuantBNConv2d",
    "QuantConv2d",
]
