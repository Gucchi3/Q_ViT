from .eformer_v1 import (
    EfficientFormer,
    eformer_v1_l1,
    eformer_v1_l3,
    eformer_v1_l7,
)

MODEL_MAP = {
    "eformer_v1_l1": eformer_v1_l1,
    "eformer_v1_l3": eformer_v1_l3,
    "eformer_v1_l7": eformer_v1_l7,
}


def get_model(name: str):
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"eformer_v1: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]
