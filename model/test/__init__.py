from .test_model import TestModel, test

MODEL_MAP = {
    "test": test,
}


def get_model(name: str):
    name_lower = name.lower()
    if name_lower not in MODEL_MAP:
        raise KeyError(
            f"test: unknown model '{name}'. "
            f"Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name_lower]
