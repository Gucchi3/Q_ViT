"""
utils.py
I-ViT 学習ユーティリティ
  - class tools       : 設定初期化・データロード・保存系
  - class TrainUtils  : 学習ループ・評価・クラス別精度
  - freeze_model / unfreeze_model : 量子化レンジ固定/解除
"""

import pretty_errors
import os
import json
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from timm.data.mixup import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import NativeScaler, ModelEma
from torcheval.metrics import MulticlassAccuracy

from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeRemainingColumn, TransferSpeedColumn,
)
from rich.console import Console
from rich.table import Table


# ── 量子化レンジ固定（I-ViT / model_utils.py 由来） ───────────────────────────────────
def freeze_model(model: nn.Module):
    """
    QuantAct の活性化レンジを固定する（再帰的）。
    running_stat 属性をもつモジュールを QuantAct とみなしてダックタイピングで判定する。
    """
    if hasattr(model, 'running_stat') and hasattr(model, 'fix'):
        model.fix()
        return
    for child in model.children():
        freeze_model(child)


def unfreeze_model(model: nn.Module):
    """
    QuantAct の活性化レンジ固定を解除する（再帰的）。
    """
    if hasattr(model, 'running_stat') and hasattr(model, 'unfix'):
        model.unfix()
        return
    for child in model.children():
        unfreeze_model(child)


# ══════════════════════════════════════════════════════════════════════════════
class tools:
    """汎用ユーティリティ"""

    # ── 初期設定 ───────────────────────────────────────────────────────────────
    @staticmethod
    def init_setting(config_path: str):
        """デバイス設定・シード固定・config 読み込み・ログフォルダ作成"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        seed = config["SEED"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        import warnings
        warnings.filterwarnings('ignore')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(config["LOG_DIR"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config["LOG_DIR"], timestamp)
        os.makedirs(run_dir, exist_ok=True)
        config["RUN_DIR"] = run_dir

        return device, config

    # ── 重み読み込み ───────────────────────────────────────────────────────────
    @staticmethod
    def load_weight(model: nn.Module, device: torch.device, config: dict) -> nn.Module:
        """保存済み重みの読み込み（LOAD_WEIGHT=1 のとき有効）"""
        if not config.get("LOAD_WEIGHT", 0):
            return model
        weight_path = config.get("LOAD_WEIGHT_PATH", "")
        if not weight_path or not os.path.exists(weight_path):
            print(f"[Warning] weight file not found: {weight_path}")
            return model
        try:
            obj = torch.load(weight_path, map_location=device, weights_only=True)
            # main.py が保存するフルチェックポイント形式にも対応
            state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded weights from {weight_path}")
        except Exception as e:
            print(f"[Error] Failed to load weights — {e}")
        return model

    # ── チェックポイントから完全リジューム ─────────────────────────────────────
    @staticmethod
    def load_checkpoint(path: str, model: nn.Module, optimizer, lr_scheduler,
                        device: torch.device) -> int:
        """
        フルチェックポイントの読み込み（epoch / model / optimizer / lr_scheduler）。
        戻り値: 次に開始すべき epoch 番号
        """
        if not path or not os.path.exists(path):
            print(f"[Warning] checkpoint not found: {path}")
            return 0
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            start_epoch = ckpt["epoch"]
            print(f"[OK] Resumed from {path}  (epoch {start_epoch})")
            return start_epoch
        except Exception as e:
            print(f"[Error] Failed to load checkpoint — {e}")
            return 0

    # ── モデル構築 ─────────────────────────────────────────────────────────────
    @staticmethod
    def build_model(config: dict, device: torch.device) -> nn.Module:
        """config の MODEL キーからモデルを生成して device に転送する"""
        model_name = config["MODEL"].lower()
        model_type = model_name.split("_")[0]   # "deit" / "swin"

        if model_type == "deit":
            from model.i_vit import get_model
        elif model_type == "swin":
            from model.swin import get_model
        else:
            raise ValueError(
                f"Unknown model type '{model_type}' derived from MODEL='{config['MODEL']}'. "
                "Supported prefixes: 'deit', 'swin'.")

        model = get_model(model_name)(
            pretrained     = config["PRETRAINED"],
            num_classes    = config["NUM_CLASSES"],
            in_chans       = config.get("IN_CHANS", 3),
            drop_rate      = config["DROP"],
            drop_path_rate = config["DROP_PATH"],
        )
        model.to(device)
        return model

    # ── 学習設定の表示 ─────────────────────────────────────────────────────────
    @staticmethod
    def print_training_info(model: nn.Module, device: torch.device, config: dict):
        """学習開始時の設定情報を rich テーブルで表示"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        table = Table(title="Training Configuration", show_header=True, header_style="bold")
        table.add_column("Section",  style="cyan", no_wrap=True)
        table.add_column("Key",      style="green")
        table.add_column("Value",    style="white")

        # 環境
        table.add_row("Environment", "Device", f"{device} ({gpu_name})")
        table.add_row("Environment", "PyTorch", torch.__version__)
        table.add_row("Environment", "Seed", str(config["SEED"]))
        table.add_row("")
        # モデル
        table.add_row("Model", "Name",        config["MODEL"])
        table.add_row("Model", "Pretrained",  str(config["PRETRAINED"]))
        table.add_row("Model", "Image Size",  str(config["IMG_SIZE"]))
        table.add_row("Model", "Num Classes", str(config["NUM_CLASSES"]))
        table.add_row("Model", "Drop",        str(config["DROP"]))
        table.add_row("Model", "Drop Path",   str(config["DROP_PATH"]))
        table.add_row("Model", "Total Params",    f"{total_params:,} ({total_params/1e6:.2f}M)")
        table.add_row("Model", "Trainable Params", f"{trainable_params:,}")
        table.add_row("")
        # データ
        table.add_row("Data", "Dataset",     config["DATA_SET"])
        table.add_row("Data", "Data Dir",    config["DATA_DIR"])
        table.add_row("")
        # 学習
        table.add_row("Training", "Epochs",      str(config["EPOCHS"]))
        table.add_row("Training", "Batch Size",   str(config["BATCH_SIZE"]))
        table.add_row("Training", "Optimizer",    config["OPTIMIZER"].upper())
        table.add_row("Training", "LR",           str(config["LEARNING_RATE"]))
        table.add_row("Training", "Weight Decay", str(config["WEIGHT_DECAY"]))
        table.add_row("Training", "Scheduler",    config["SCHEDULER"])
        table.add_row("Training", "Warmup Epochs",str(config["WARMUP_EPOCHS"]))
        table.add_row("Training", "LR Min",       str(config["LR_MIN"]))
        table.add_row("Training", "Model EMA",    str(config["MODEL_EMA"]))
        table.add_row("")
        # データ拡張
        table.add_row("Augmentation", "Mixup α",      str(config["MIXUP_ALPHA"]))
        table.add_row("Augmentation", "CutMix α",     str(config["CUTMIX_ALPHA"]))
        table.add_row("Augmentation", "Label Smooth", str(config["LABEL_SMOOTHING"]))
        table.add_row("Augmentation", "AutoAugment",  str(config["AUTO_AUGMENT"]))
        table.add_row("")
        # 出力
        table.add_row("Output", "Run Directory", config["RUN_DIR"])

        console = Console()
        console.print()
        console.print(table)
        console.print()

    # ── データローダー作成 ─────────────────────────────────────────────────────
    @staticmethod
    def make_dataloader(config: dict):
        """
        DataLoader を作成する。
        DATA_SET: CIFAR10 / CIFAR100 / IMNET
        戻り値: (train_loader, test_loader, classes)
                classes は IMNET のとき None を返す。
        """
        dataset = config["DATA_SET"].upper()
        if dataset in ("CIFAR10", "CIFAR"):
            return tools._make_loader_cifar(config, 10)
        elif dataset == "CIFAR100":
            return tools._make_loader_cifar(config, 100)
        elif dataset in ("IMNET", "IMAGENET"):
            return tools._make_loader_imagenet(config)
        else:
            raise ValueError(f"Unknown DATA_SET: {config['DATA_SET']}")

    @staticmethod
    def _make_loader_cifar(config: dict, n_classes: int):
        """CIFAR-10 / CIFAR-100 DataLoader"""
        if n_classes == 10:
            MEAN = (0.4914, 0.4822, 0.4465)
            STD  = (0.2470, 0.2435, 0.2616)
        else:
            MEAN = (0.5071, 0.4867, 0.4408)
            STD  = (0.2675, 0.2565, 0.2761)

        auto_aug = config.get("AUTO_AUGMENT", None)
        if auto_aug in ("none", "None", "", None):
            auto_aug = None

        img_size = config["IMG_SIZE"]
        use_resize = img_size > 32  # 224 のときはリサイズ、32 のときは RandomCrop

        train_transform = create_transform(
            input_size=(config["IN_CHANS"], img_size, img_size),
            is_training=True,
            color_jitter=config.get("COLOR_JITTER", 0.4),
            auto_augment=auto_aug,
            interpolation=config.get("INTERPOLATION", "bicubic"),
            mean=MEAN,
            std=STD,
            normalize=True,
            re_prob=config.get("RE_PROB", 0.25),
            re_mode=config.get("RE_MODE", "pixel"),
            re_count=config.get("RE_COUNT", 1),
        )
        # img_size <= 32 のとき RandomResizedCrop → RandomCrop に差し替え
        if not use_resize:
            train_transform.transforms[0] = transforms.RandomCrop(img_size, padding=4)

        test_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        data_dir = config.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
        Dataset = torchvision.datasets.CIFAR10 if n_classes == 10 else torchvision.datasets.CIFAR100
        train_set = Dataset(root=data_dir, train=True,  download=True,  transform=train_transform)
        test_set  = Dataset(root=data_dir, train=False, download=True,  transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config["BATCH_SIZE"], shuffle=True,
            num_workers=config["NUM_WORKERS"], pin_memory=config.get("PIN_MEM", True),
            drop_last=True, persistent_workers=(config["NUM_WORKERS"] > 0))
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=config["BATCH_SIZE"], shuffle=False,
            num_workers=config["NUM_WORKERS"], pin_memory=config.get("PIN_MEM", True),
            drop_last=False, persistent_workers=(config["NUM_WORKERS"] > 0))

        if n_classes == 10:
            classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            classes = None  # CIFAR-100 はクラス名省略（多すぎるため）

        return train_loader, test_loader, classes

    @staticmethod
    def _make_loader_imagenet(config: dict):
        """ImageNet DataLoader"""
        img_size = config["IMG_SIZE"]
        auto_aug = config.get("AUTO_AUGMENT", None)
        if auto_aug in ("none", "None", "", None):
            auto_aug = None

        train_transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=config.get("COLOR_JITTER", 0.4),
            auto_augment=auto_aug,
            interpolation=config.get("INTERPOLATION", "bicubic"),
            re_prob=config.get("RE_PROB", 0.25),
            re_mode=config.get("RE_MODE", "pixel"),
            re_count=config.get("RE_COUNT", 1),
        )
        size = int((256 / 224) * img_size)
        test_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

        data_dir = config["DATA_DIR"]
        train_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
        test_set  = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config["BATCH_SIZE"], shuffle=True,
            num_workers=config["NUM_WORKERS"], pin_memory=config.get("PIN_MEM", True),
            drop_last=True, persistent_workers=(config["NUM_WORKERS"] > 0))
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=config["BATCH_SIZE"], shuffle=False,
            num_workers=config["NUM_WORKERS"], pin_memory=config.get("PIN_MEM", True),
            drop_last=False, persistent_workers=(config["NUM_WORKERS"] > 0))

        return train_loader, test_loader, None

    # ── モデル保存 ─────────────────────────────────────────────────────────────
    @staticmethod
    def save_model(model: nn.Module, config: dict, filename: str = "model.pth"):
        """モデルの重みを .pth 形式で保存"""
        try:
            path = os.path.join(config["RUN_DIR"], filename)
            torch.save(model.state_dict(), path)
            print(f"[OK] Model saved at {path}")
        except Exception as e:
            print(f"[Error] Failed to save model — {e}")

    # ── グラフ保存 ─────────────────────────────────────────────────────────────
    @staticmethod
    def save_curves(train_losses, test_losses, test_accs, config: dict):
        """学習曲線をプロットして保存"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(train_losses, label="train"); ax1.plot(test_losses, label="test")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.grid()
            ax2.plot([a * 100 for a in test_accs], label="test acc", color="r")
            ax2.set_xlabel("epoch"); ax2.set_ylabel("acc (%)"); ax2.legend(); ax2.grid()
            path = os.path.join(config["RUN_DIR"], "curves.png")
            fig.savefig(path); plt.close(fig)
            print(f"[OK] Curves saved at {path}")
        except Exception as e:
            print(f"[Error] Failed to save curves — {e}")

    # ── 学習情報保存 ───────────────────────────────────────────────────────────
    @staticmethod
    def save_training_info(model: nn.Module, device: torch.device, config: dict):
        """学習設定・結果をJSON形式で保存"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        info = {
            "Environment": {
                "Device": f"{device} ({gpu_name})",
                "PyTorch": torch.__version__,
                "Seed": config["SEED"]
            },
            "Model": {
                "Name":            config["MODEL"],
                "Pretrained":      config["PRETRAINED"],
                "Image Size":      config["IMG_SIZE"],
                "In Channels":     config["IN_CHANS"],
                "Num Classes":     config["NUM_CLASSES"],
                "Drop":            config["DROP"],
                "Drop Path":       config["DROP_PATH"],
                "Total Params":    f"{total_params:,} ({total_params/1e6:.2f}M)",
                "Trainable":       f"{trainable_params:,}"
            },
            "Dataset": {
                "Dataset":         config["DATA_SET"],
                "Data Dir":        config["DATA_DIR"]
            },
            "Training": {
                "Epochs":          config["EPOCHS"],
                "Start Epoch":     config.get("START_EPOCH", 0),
                "Batch Size":      config["BATCH_SIZE"],
                "Num Workers":     config["NUM_WORKERS"],
                "Pin Memory":      config.get("PIN_MEM", True),
                "Clip Grad":       config.get("CLIP_GRAD")
            },
            "Optimizer": {
                "Optimizer":       config["OPTIMIZER"].upper(),
                "LR":              config["LEARNING_RATE"],
                "Weight Decay":    config["WEIGHT_DECAY"],
                "Momentum":        config.get("MOMENTUM", 0.9),
                "Eps":             config.get("OPTIMIZER_EPS", 1e-8),
                "Betas":           config.get("OPTIMIZER_BETAS", [0.9, 0.999])
            },
            "Scheduler": {
                "Scheduler":       config["SCHEDULER"],
                "LR Min":          config["LR_MIN"],
                "Warmup Epochs":   config["WARMUP_EPOCHS"],
                "Warmup LR Init":  config.get("WARMUP_LR_INIT", 1e-6),
                "Warmup Prefix":   config.get("WARMUP_PREFIX", True),
                "Decay Epochs":    config.get("DECAY_EPOCHS", 30),
                "Cooldown Epochs": config.get("COOLDOWN_EPOCHS", 10),
                "Patience Epochs": config.get("PATIENCE_EPOCHS", 10),
                "Decay Rate":      config.get("DECAY_RATE", 0.1)
            },
            "Augmentation": {
                "Mixup α":         config["MIXUP_ALPHA"],
                "CutMix α":        config["CUTMIX_ALPHA"],
                "CutMix MinMax":   config.get("CUTMIX_MINMAX"),
                "Mixup Prob":      config.get("MIXUP_PROB", 1.0),
                "Mixup Switch Prob": config.get("MIXUP_SWITCH_PROB", 0.5),
                "Mixup Mode":      config.get("MIXUP_MODE", "batch"),
                "Label Smooth":    config["LABEL_SMOOTHING"],
                "Color Jitter":    config.get("COLOR_JITTER", 0.4),
                "AutoAugment":     config["AUTO_AUGMENT"],
                "Interpolation":   config.get("INTERPOLATION", "bicubic"),
                "RE Prob":         config.get("RE_PROB", 0.25),
                "RE Mode":         config.get("RE_MODE", "pixel"),
                "RE Count":        config.get("RE_COUNT", 1),
                "RE Split":        config.get("RE_SPLIT", False)
            },
            "EMA": {
                "Model EMA":       config.get("MODEL_EMA", False),
                "EMA Decay":       config.get("MODEL_EMA_DECAY", 0.99996),
                "EMA Force CPU":   config.get("MODEL_EMA_FORCE_CPU", False)
            },
            "Load Weight": {
                "Load Weight":     config.get("LOAD_WEIGHT", 0),
                "Weight Path":     config.get("LOAD_WEIGHT_PATH", "")
            },
            "Output": {
                "Log Dir":         config.get("LOG_DIR", "./log/"),
                "Run Directory":   config["RUN_DIR"]
            }
        }

        path = os.path.join(config["RUN_DIR"], "training_info.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
            print(f"[OK] Training info saved at {path}")
        except Exception as e:
            print(f"[Error] Failed to save training info — {e}")


# ══════════════════════════════════════════════════════════════════════════════
class TrainUtils:
    """学習ループ・評価系ユーティリティ"""

    # ── 学習コンポーネント一括初期化 ──────────────────────────────────────────
    @staticmethod
    def setup_training(model: nn.Module, config: dict, device: torch.device):
        """
        optimizer / scheduler / criterion / mixup / scaler / ema をまとめて初期化する。
        戻り値: (mixup_fn, train_criterion, test_criterion, optimizer,
                 lr_scheduler, loss_scaler, model_ema, metric, epochs)
        """
        # ── Mixup /CutMix ──────────────────────────────────────────────────
        mixup_active = (config["MIXUP_ALPHA"] > 0 or config["CUTMIX_ALPHA"] > 0 or config.get("CUTMIX_MINMAX") is not None)
        
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=config["MIXUP_ALPHA"],
                cutmix_alpha=config["CUTMIX_ALPHA"],
                cutmix_minmax=config.get("CUTMIX_MINMAX"),
                prob=config["MIXUP_PROB"],
                switch_prob=config["MIXUP_SWITCH_PROB"],
                mode=config["MIXUP_MODE"],
                label_smoothing=config["LABEL_SMOOTHING"],
                num_classes=config["NUM_CLASSES"])
            train_criterion = SoftTargetCrossEntropy()
        else:
            mixup_fn = None
            train_criterion = LabelSmoothingCrossEntropy(smoothing=config["LABEL_SMOOTHING"]) if config["LABEL_SMOOTHING"] > 0 else nn.CrossEntropyLoss()

        test_criterion = nn.CrossEntropyLoss()

        # ── オプティマイザ（timm create_optimizer_v2 を使用） ──────────────
        # timm の create_optimizer_v2 は args オブジェクトの代わりに kwargs を受け付ける
        optimizer = create_optimizer_v2(
            model,
            opt=config["OPTIMIZER"],
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
            momentum=config.get("MOMENTUM", 0.9),
            eps=config.get("OPTIMIZER_EPS", 1e-8),
            betas=tuple(config.get("OPTIMIZER_BETAS", [0.9, 0.999])),
        )

        # ── スケジューラ ────────────────────────────────────────────────────
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config["EPOCHS"],
            lr_min=config["LR_MIN"],
            warmup_t=config["WARMUP_EPOCHS"],
            warmup_lr_init=config["WARMUP_LR_INIT"],
            warmup_prefix=config.get("WARMUP_PREFIX", True),
        )

        # ── AMP スケーラー ──────────────────────────────────────────────────
        loss_scaler = NativeScaler()

        # ── Model EMA ──────────────────────────────────────────────────────
        model_ema = None
        if config.get("MODEL_EMA", False):
            model_ema = ModelEma(
                model,
                decay=config.get("MODEL_EMA_DECAY", 0.99996),
                device='cpu' if config.get("MODEL_EMA_FORCE_CPU", False) else '',
                resume='')

        # ── 評価指標 ────────────────────────────────────────────────────────
        metric = MulticlassAccuracy(num_classes=config["NUM_CLASSES"], device=device)
        epochs = config["EPOCHS"]

        return mixup_fn, train_criterion, test_criterion, optimizer, \
               lr_scheduler, loss_scaler, model_ema, metric, epochs

    # ── 1エポック学習 ──────────────────────────────────────────────────────────
    @staticmethod
    def train_one_epoch(model, device, loader, mixup_fn, criterion, optimizer,
                        loss_scaler: NativeScaler, clip_grad, model_ema):
        """1エポック分の学習（NativeScaler AMP + rich progress bar）"""
        model.train()
        unfreeze_model(model)

        loss_sum, count = 0.0, 0

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="grey30", complete_style="bold green", finished_style="bold green"),
            MofNCompleteColumn(), TimeRemainingColumn(), TransferSpeedColumn(),
            transient=True, expand=True, redirect_stdout=True, redirect_stderr=True,
        ) as progress:
            
            task = progress.add_task("Train", total=len(loader))
            for i, (data, target) in enumerate(loader):
                data   = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if mixup_fn is not None:
                    data, target = mixup_fn(data, target)

                output = model(data)
                loss   = criterion(output, target)

                optimizer.zero_grad()
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters(), create_graph=is_second_order)
                torch.cuda.synchronize()

                if model_ema is not None:
                    model_ema.update(model)

                loss_sum += loss.item() * data.size(0)
                count    += data.size(0)
                progress.update(task, advance=1)

        return loss_sum / count

    # ── 評価 ──────────────────────────────────────────────────────────────────
    @staticmethod
    @torch.no_grad()
    def evaluate(model, device, loader, criterion, metric):
        """テストデータでの評価"""
        model.eval()
        freeze_model(model)

        loss_sum, count = 0.0, 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="grey30",
                      complete_style="bold green", finished_style="bold green"),
            MofNCompleteColumn(), TimeRemainingColumn(), TransferSpeedColumn(),
            transient=True, expand=True, redirect_stdout=True, redirect_stderr=True,
        ) as progress:
            task = progress.add_task("Test", total=len(loader))
            for data, target in loader:
                data   = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                loss   = criterion(output, target)
                loss_sum += loss.item() * data.size(0)
                count    += data.size(0)
                metric.update(output, target)
                progress.update(task, advance=1)

        acc = metric.compute().item()
        metric.reset()
        return loss_sum / count, acc

    # ── クラス別正答率 ─────────────────────────────────────────────────────────
    @staticmethod
    def class_accuracy(model, device, loader, classes):
        """クラス別正答率を計算して表示（classes が None のときはスキップ）"""
        if classes is None:
            return
        model.eval()
        correct_pred = {c: 0 for c in classes}
        total_pred   = {c: 0 for c in classes}
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                for lbl, pred in zip(labels, predicted):
                    if lbl == pred:
                        correct_pred[classes[lbl.item()]] += 1
                    total_pred[classes[lbl.item()]] += 1
        print("\nClass Accuracy:")
        for cls, cnt in correct_pred.items():
            total = total_pred[cls]
            print(f"  {cls:>10s}: {100*cnt/total:.1f}%  ({cnt}/{total})")
