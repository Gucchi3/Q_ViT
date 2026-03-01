from .eformer_v2 import (
    EfficientFormerV2,
    eformer_v2_s0,
    eformer_v2_s1,
    eformer_v2_s2,
    eformer_v2_l,
)

MODEL_MAP = {
    "eformer_v2_s0": eformer_v2_s0,
    "eformer_v2_s1": eformer_v2_s1,
    "eformer_v2_s2": eformer_v2_s2,
    "eformer_v2_l":  eformer_v2_l,
}


def get_model(name: str):
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"eformer_v2: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]
