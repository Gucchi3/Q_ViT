from .tiny_eformer_model import TinyEFormer, tiny_eformer

MODEL_MAP = {
    "tiny_eformer": tiny_eformer,
}


def get_model(name: str):
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"tiny_eformer: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]
