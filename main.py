"""
main.py
I-ViT 量子化 ViT/Swin 学習エントリポイント

Usage:
    python main.py [config.json のパス]  (省略時は ./config.json) -> python main.py
    #! バックグラウンドで実行する場合は、「nohup python -u main.py > train.log 2>&1 &」
"""
# ── import ───────────────────────────────────────────────────────────────────
import pretty_errors
import os
import sys
import datetime

import torch
import torch.nn as nn

from timm.utils import NativeScaler, ModelEma

from utils import tools, TrainUtils, freeze_model, unfreeze_model

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # ── 設定読み込み ──────────────────────────────────────────────────────────
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    device, config = tools.init_setting(config_path)
    run_dir = config["RUN_DIR"]

    # ── モデル構築 ────────────────────────────────────────────────────────────
    model = tools.build_model(config, device)

    # ── 事前学習重みの読み込み（START_EPOCH==0 のとき＝新規学習のみ） ──────────
    if (config.get("LOAD_WEIGHT", 0) and config.get("LOAD_WEIGHT_PATH", "")
            and config.get("START_EPOCH", 0) == 0):
        tools.load_weight(model, device, config)

    # ── データローダー ────────────────────────────────────────────────────────
    train_loader, test_loader, classes = tools.make_dataloader(config)

    # ── 学習設定表示 ──────────────────────────────────────────────────────────
    tools.print_training_info(model, device, config)
    tools.save_training_info(model, device, config)

    # ── 学習コンポーネント初期化 ───────────────────────────────────────────────
    (mixup_fn ,  train_criterion,  test_criterion,
     optimizer,  lr_scheduler   ,  loss_scaler   ,
     model_ema,  metric         ,  epochs         ) = TrainUtils.setup_training(model, config, device)

    start_epoch  = config.get("START_EPOCH", 0)
    clip_grad    = config.get("CLIP_GRAD", None)
    best_acc1    = 0.0

    # START_EPOCH > 0 のときはフルチェックポイントからリジューム
    # START_EPOCH == 0 のときは事前学習重みのみ読み込み（load_weight は main 冒頭で実行済み）
    if start_epoch > 0 and config.get("LOAD_WEIGHT", 0) and config.get("LOAD_WEIGHT_PATH", ""):
        start_epoch = tools.load_checkpoint(
            config["LOAD_WEIGHT_PATH"], model, optimizer, lr_scheduler, device)

    train_losses, test_losses, test_accs = [], [], []

    # ── 学習ループ ────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, epochs):
        # スケジューラを epoch 先頭で更新
        lr_scheduler.step(epoch)

        train_loss = TrainUtils.train_one_epoch(model, device, train_loader, mixup_fn, train_criterion,
                                                optimizer, loss_scaler, clip_grad, model_ema)

        test_loss, test_acc = TrainUtils.evaluate(model, device, test_loader, test_criterion, metric)

        # ModelEMA でも評価
        ema_test_acc = None
        if model_ema is not None:
            _, ema_test_acc = TrainUtils.evaluate(model_ema.ema, device, test_loader, test_criterion, metric)

        # ── ログ表示 ──────────────────────────────────────────────────────
        lr_now = optimizer.param_groups[0]["lr"]
        acc_str = f"acc={test_acc*100:.2f}%"
        if ema_test_acc is not None:
            acc_str += f"  ema_acc={ema_test_acc*100:.2f}%"
        print(f"Epoch [{epoch+1:3d}/{epochs}]  "
              f"lr={lr_now:.2e}  "
              f"train_loss={train_loss:.4f}  "
              f"test_loss={test_loss:.4f}  "
              f"{acc_str}")

        # ── ベストモデル更新 ───────────────────────────────────────────────
        current_acc = ema_test_acc if ema_test_acc is not None else test_acc
        if current_acc > best_acc1:
            best_acc1 = current_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc1": best_acc1,
                "config": config,
            }
            if model_ema is not None:
                checkpoint["model_ema"] = model_ema.ema.state_dict()
            ckpt_path = os.path.join(run_dir, "checkpoint_best.pth.tar")
            torch.save(checkpoint, ckpt_path)
            # print(f"  --> Best model updated: acc={best_acc1*100:.2f}%  path={ckpt_path}")

        # ── 最新チェックポイント（resume 用） ─────────────────────────────
        latest_ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_acc1": best_acc1,
            "config": config,
        }
        torch.save(latest_ckpt, os.path.join(run_dir, "checkpoint_latest.pth.tar"))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # ── 学習曲線 ─────────────────────────────────────────────────────
        tools.save_curves(train_losses, test_losses, test_accs, config)

    # ── 学習終了 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training finished — best acc1 = {best_acc1*100:.2f}%")
    print(f"{'='*60}\n")

    # ── クラス別精度 ─────────────────────────────────────────────────────
    TrainUtils.class_accuracy(model, device, test_loader, classes)

    # ── 最終モデル保存 ───────────────────────────────────────────────────
    tools.save_model(model, config, filename="model_final.pth")




if __name__ == "__main__":
    main()
