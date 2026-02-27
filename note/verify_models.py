"""
verify_models.py
torchinfo を使って test/ モデルが I-ViT 原本と等価であることを確認するスクリプト。

比較内容:
  1. 総パラメータ数が一致するか
  2. 各レイヤーの名前・形状が一致するか（state_dict キー比較）
  3. VisionTransformer / SwinTransformer 双方の Tiny / Small を検証

Usage:
    cd /home/ihpc/yamaguchi/test
    python verify_models.py
"""

import pretty_errors
import sys
import os

# ── パス設定 ──────────────────────────────────────────────────────────────────
TEST_DIR  = os.path.dirname(os.path.abspath(__file__))
IVIT_DIR  = os.path.join(os.path.dirname(TEST_DIR), "I-ViT")

# ── torchinfo インポート確認 ──────────────────────────────────────────────────
try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("[Warning] torchinfo not installed — skipping column-level summary.")
    print("          Install: pip install torchinfo\n")

import torch
from functools import partial

# ─────────────────────────────────────────────────────────────────────────────
def compare_state_dicts(name_a: str, sd_a: dict, name_b: str, sd_b: dict) -> bool:
    """2 つの state_dict のキー・テンソル形状を比較する。"""
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    ok = True

    if only_a:
        print(f"  [DIFF] keys only in {name_a}:")
        for k in sorted(only_a): print(f"    {k}")
        ok = False
    if only_b:
        print(f"  [DIFF] keys only in {name_b}:")
        for k in sorted(only_b): print(f"    {k}")
        ok = False

    shape_mismatch = []
    for k in keys_a & keys_b:
        if sd_a[k].shape != sd_b[k].shape:
            shape_mismatch.append((k, sd_a[k].shape, sd_b[k].shape))
    if shape_mismatch:
        print(f"  [DIFF] shape mismatches ({len(shape_mismatch)} keys):")
        for k, sa, sb in shape_mismatch:
            print(f"    {k}: {name_a}={sa}, {name_b}={sb}")
        ok = False

    return ok


def run_torchinfo(model, input_size, model_name: str):
    if not HAS_TORCHINFO:
        return
    x     = torch.zeros(1, *input_size)
    quant = torch.tensor(1.0)
    print(f"\n  [torchinfo] {model_name}")
    print("  " + "-"*60)
    try:
        result = summary(model, input_data=[x, quant], depth=2,
                         col_names=["input_size","output_size","num_params"],
                         verbose=0)
        total = result.total_params
        trainable = result.trainable_params
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}")
    except Exception:
        # 量子化モデルの forward() は act_scaling_factor を追加引数に取るため
        # torchinfo の自動トレースが失敗する。手動カウントで代替。
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params    : {total:,}  (torchinfo 非対応: 手動計上)")
        print(f"  Trainable params: {trainable:,}")


# ─────────────────────────────────────────────────────────────────────────────
# DeiT (VisionTransformer) の検証
# ─────────────────────────────────────────────────────────────────────────────
def verify_deit():
    print("="*65)
    print(" DeiT (VisionTransformer) 等価性検証")
    print("="*65)

    # ── 原本モデルのロード ────────────────────────────────────────────────
    sys.path.insert(0, IVIT_DIR)
    from models.vit_quant import deit_tiny_patch16_224 as orig_tiny
    from models.vit_quant import deit_small_patch16_224 as orig_small

    # ── test モデルのロード ───────────────────────────────────────────────
    sys.path.insert(0, TEST_DIR)
    from model.i_vit import get_model
    test_tiny_fn  = get_model("deit_tiny")
    test_small_fn = get_model("deit_small")

    for variant, orig_fn, test_fn in [
        ("DeiT-Tiny",  orig_tiny,  test_tiny_fn),
        ("DeiT-Small", orig_small, test_small_fn),
    ]:
        print(f"\n── {variant} ──")
        orig_model = orig_fn(pretrained=False, num_classes=1000)
        test_model = test_fn(pretrained=False, num_classes=1000)

        orig_sd = orig_model.state_dict()
        test_sd = test_model.state_dict()

        total_orig = sum(p.numel() for p in orig_model.parameters())
        total_test = sum(p.numel() for p in test_model.parameters())
        params_match = (total_orig == total_test)
        print(f"  Original params : {total_orig:,}")
        print(f"  Test     params : {total_test:,}")
        print(f"  Param count     : {'OK ✓' if params_match else 'MISMATCH ✗'}")

        sd_ok = compare_state_dicts("original", orig_sd, "test", test_sd)
        if sd_ok:
            print("  state_dict keys : OK ✓ (全キー・形状一致)")
        else:
            print("  state_dict keys : MISMATCH ✗")

        run_torchinfo(test_model, (3, 224, 224), variant)

    # パス汚染防止
    sys.path.pop(sys.path.index(IVIT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Swin Transformer の検証
# ─────────────────────────────────────────────────────────────────────────────
def verify_swin():
    print("\n" + "="*65)
    print(" Swin Transformer 等価性検証")
    print("="*65)

    sys.path.insert(0, IVIT_DIR)
    from models.swin_quant import swin_tiny_patch4_window7_224 as orig_tiny
    from models.swin_quant import swin_small_patch4_window7_224 as orig_small

    sys.path.insert(0, TEST_DIR)
    from model.swin import get_model
    test_tiny_fn  = get_model("swin_tiny")
    test_small_fn = get_model("swin_small")

    for variant, orig_fn, test_fn in [
        ("Swin-Tiny",  orig_tiny,  test_tiny_fn),
        ("Swin-Small", orig_small, test_small_fn),
    ]:
        print(f"\n── {variant} ──")
        orig_model = orig_fn(pretrained=False, num_classes=1000)
        test_model = test_fn(pretrained=False, num_classes=1000)

        total_orig = sum(p.numel() for p in orig_model.parameters())
        total_test = sum(p.numel() for p in test_model.parameters())
        params_match = (total_orig == total_test)
        print(f"  Original params : {total_orig:,}")
        print(f"  Test     params : {total_test:,}")
        print(f"  Param count     : {'OK ✓' if params_match else 'MISMATCH ✗'}")

        orig_sd = orig_model.state_dict()
        test_sd = test_model.state_dict()
        sd_ok = compare_state_dicts("original", orig_sd, "test", test_sd)
        if sd_ok:
            print("  state_dict keys : OK ✓ (全キー・形状一致)")
        else:
            print("  state_dict keys : MISMATCH ✗")

        run_torchinfo(test_model, (3, 224, 224), variant)

    sys.path.pop(sys.path.index(IVIT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verify_deit()
    verify_swin()
    print("\n" + "="*65)
    print(" 検証完了")
    print("="*65)
