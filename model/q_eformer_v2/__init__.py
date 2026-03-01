from .q_eformer_v2 import (
    Q_EfficientFormerV2,
    q_eformer_v2_s0,
    q_eformer_v2_s1,
    q_eformer_v2_s2,
    q_eformer_v2_l,
)

MODEL_MAP = {
    "q_eformer_v2_s0": q_eformer_v2_s0,
    "q_eformer_v2_s1": q_eformer_v2_s1,
    "q_eformer_v2_s2": q_eformer_v2_s2,
    "q_eformer_v2_l":  q_eformer_v2_l,
}


def get_model(name: str):
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"q_eformer_v2: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]
