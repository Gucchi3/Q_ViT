from .test2_model import Test2Model, test2

MODEL_MAP = {
    "test2": test2,
}


def get_model(name: str):
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"test2: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]
