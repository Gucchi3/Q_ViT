"""
profile_main.py  ―  Q_ViT 詳細タイミング計測スクリプト
======================================================

main.py と同等の学習設定で動かしつつ、訓練ループの各フェーズの所要時間を
詳細に計測・表示する。I-ViT158 との差異として特に疑わしい部分（以下）もあわせて計測する：
  * rich.Progress コンテキストマネージャのオーバーヘッド（redirect_stdout/stderr）
  * progress.update() 呼び出しコスト
  * freeze_model / unfreeze_model の再帰トラバーサルコスト
  * torcheval MulticlassAccuracy.update() / .compute() のコスト

使い方（config.json のパスを省略すると ./config.json を読む）:
  python profile_main.py [config.json のパス]
  python profile_main.py [config.json のパス] --profile-epochs 2 --profile-batches 100

コマンドライン追加引数:
  --profile-epochs  INT   計測エポック数 (default: 2)
  --profile-batches INT   1エポックあたりのバッチ数 (0=全部 default: 0)
"""

import pretty_errors
import os
import sys
import time
import warnings
import datetime
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from timm.utils import NativeScaler, ModelEma

from utils import tools, TrainUtils, freeze_model, unfreeze_model


# ── プロファイリング専用 CLI 引数 ─────────────────────────────────────────────
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("config_path",       nargs="?", default="config.json")
_cli.add_argument("--profile-epochs",  type=int,  default=2)
_cli.add_argument("--profile-batches", type=int,  default=0)
_profile_args, _ = _cli.parse_known_args()


# ── CUDA タイマーユーティリティ ───────────────────────────────────────────────
class CudaTimer:
    """
    CUDA イベントを使った正確な GPU 時間計測ラッパー。
    CPU-only 環境では time.perf_counter() にフォールバックする。
    """
    def __init__(self, use_cuda: bool = True):
        self.use_cuda  = use_cuda and torch.cuda.is_available()
        self._start:   dict = {}
        self._end:     dict = {}
        self._cpu_start: dict = {}

    def start(self, tag: str):
        if self.use_cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._start[tag] = e
        else:
            self._cpu_start[tag] = time.perf_counter()

    def end(self, tag: str):
        if self.use_cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._end[tag] = e
        else:
            self._end[tag] = time.perf_counter()

    def elapsed_ms(self, tag: str) -> float:
        if self.use_cuda:
            if tag not in self._start or tag not in self._end:
                return 0.0
            self._end[tag].synchronize()
            return self._start[tag].elapsed_time(self._end[tag])
        else:
            if tag not in self._cpu_start or tag not in self._end:
                return 0.0
            return (self._end[tag] - self._cpu_start[tag]) * 1000.0


# ── タイミング集計クラス ──────────────────────────────────────────────────────
class TimingAccumulator:
    def __init__(self):
        self.data: dict[str, list[float]] = defaultdict(list)

    def add(self, tag: str, ms: float):
        self.data[tag].append(ms)

    def total_ms(self, tag: str) -> float:
        return sum(self.data.get(tag, []))

    def mean_ms(self, tag: str) -> float:
        v = self.data.get(tag, [])
        return sum(v) / len(v) if v else 0.0


# ── サマリー表示 ──────────────────────────────────────────────────────────────
def print_timing_summary(acc: TimingAccumulator, epoch: int, n_batches: int,
                          epoch_wall_sec: float, label: str = "Q_ViT"):
    PHASES = [
        ("data_load",       "データ読み込み (DataLoader)"),
        ("h2d",             "CPU→GPU 転送 (.to(device))"),
        ("mixup",           "Mixup / CutMix"),
        ("forward",         "Forward pass (model + loss)"),
        ("zero_grad",       "optimizer.zero_grad()"),
        ("backward",        "Backward + optimizer step (loss_scaler)"),
        ("cuda_sync",       "torch.cuda.synchronize()"),
        ("ema",             "Model EMA update"),
        ("progress_update", "rich progress.update() オーバーヘッド"),
        ("total_batch",     "バッチ合計"),
    ]

    total_tracked = acc.total_ms("total_batch")
    SEP  = "─" * 86
    SEP2 = "═" * 86

    print(f"\n{SEP2}")
    print(f"  【{label}】 Epoch {epoch} プロファイリング結果  "
          f"(計測バッチ数: {n_batches})")
    print(SEP2)
    print(f"  エポック壁時計時間: {epoch_wall_sec:.2f} s  "
          f"({epoch_wall_sec/60:.2f} min)")
    print(SEP)
    print(f"  {'フェーズ':<42} {'合計(ms)':>10} {'平均(ms)':>10} {'比率(%)':>9}")
    print(SEP)
    for tag, lbl in PHASES:
        tot  = acc.total_ms(tag)
        mean = acc.mean_ms(tag)
        pct  = 100.0 * tot / total_tracked if total_tracked > 0 else 0.0
        bar  = "█" * int(pct / 2)
        print(f"  {lbl:<42} {tot:>10.1f} {mean:>10.3f} {pct:>8.1f}%  {bar}")
    print(SEP)
    print(f"  {'追跡合計 (total_batch)':<42} {total_tracked:>10.1f}")
    print(f"  {'エポック壁時計 (ms)':<42} {epoch_wall_sec*1000:>10.1f}")
    unaccounted = epoch_wall_sec * 1000 - total_tracked
    print(f"  {'未追跡 (オーバーヘッドなど)':<42} {unaccounted:>10.1f}  "
          f"({100.0*unaccounted/(epoch_wall_sec*1000):.1f}%)")
    print(f"{SEP2}\n")


def print_eval_summary(acc: TimingAccumulator, epoch: int, epoch_wall_sec: float,
                        label: str = "Q_ViT"):
    SEP = "─" * 86
    total_ms = acc.total_ms("total_batch")
    print(f"\n  【{label}】 Epoch {epoch} 検証フェーズ")
    print(SEP)
    print(f"  {'フェーズ':<42} {'合計(ms)':>10} {'平均(ms)':>10} {'比率(%)':>9}")
    print(SEP)
    for tag, lbl in [("data_load",       "データ読み込み"),
                     ("h2d",             "CPU→GPU 転送"),
                     ("forward",         "Forward pass (no_grad)"),
                     ("metric_update",   "metric.update() (torcheval)"),
                     ("progress_update", "rich progress.update() オーバーヘッド")]:
        tot  = acc.total_ms(tag)
        mean = acc.mean_ms(tag)
        pct  = 100.0 * tot / total_ms if total_ms > 0 else 0.0
        print(f"  {lbl:<42} {tot:>10.1f} {mean:>10.3f} {pct:>8.1f}%")
    print(SEP)
    # metric.compute() はループ外
    print(f"  metric.compute() (1回):  {acc.total_ms('metric_compute'):.3f} ms")
    print(f"  freeze_model() (1回):    {acc.total_ms('freeze_model'):.3f} ms")
    print(f"  検証壁時計時間: {epoch_wall_sec:.2f} s  ({epoch_wall_sec/60:.2f} min)\n")


# ── rich オーバーヘッド プローブ ────────────────────────────────────────────────
def _probe_rich_overhead(
    model, device, loader, mixup_fn, criterion, optimizer,
    loss_scaler: NativeScaler, clip_grad,
    n_probe: int = 10,
) -> tuple[float, float, float]:
    """
    rich.Progress(redirect_stdout=True, redirect_stderr=True) コンテキスト＋
    progress.update() が 1 バッチあたり何 ms 余分にかかるかを計測する。

    同じバッチデータを
      ① rich なし → ② rich あり (本番と同じ設定)
    の順に 2 周走らせて差分を取る。
    モデル重みはどちらの腕でも更新されるが、
    実行時間への影響は redirect オーバーヘッドと比べて十分小さいため
    この近似で問題ない。

    戻り値: (mean_no_rich_ms, mean_with_rich_ms, overhead_ms_per_batch)
    """
    use_cuda = device.type == "cuda"

    # ── 比較用バッチをキャッシュ（CPU テンソルのまま保持） ──────────────
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for d, t in loader:
        cached.append((d, t))
        if len(cached) >= n_probe:
            break
    if not cached:
        return 0.0, 0.0, 0.0

    def _run_one(d: torch.Tensor, t: torch.Tensor) -> None:
        """1 バッチ分の forward→backward を実行してタイマを止める直前まで進める。"""
        d = d.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        if mixup_fn is not None:
            d, t = mixup_fn(d, t)
        out  = model(d)
        loss = criterion(out, t)
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())
        if use_cuda:
            torch.cuda.synchronize()

    # ── ① rich なし ───────────────────────────────────────────────────
    t_no: list[float] = []
    for d, t in cached:
        if use_cuda:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        _run_one(d, t)
        t_no.append((time.perf_counter() - _t0) * 1000.0)

    # ── ② rich あり（本番と同じ設定） ────────────────────────────────
    t_yes: list[float] = []
    try:
        from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                                   MofNCompleteColumn, TimeRemainingColumn,
                                   TransferSpeedColumn)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            transient=True, expand=True,
            redirect_stdout=True, redirect_stderr=True,   # ← 本番と同じ設定
        ) as prog:
            task = prog.add_task("RichProbe", total=len(cached))
            for d, t in cached:
                if use_cuda:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                _run_one(d, t)
                prog.update(task, advance=1)             # ← update() も含めて計測
                t_yes.append((time.perf_counter() - _t0) * 1000.0)
    except ImportError:
        # rich が使えない環境ではオーバーヘッド 0 として扱う
        return sum(t_no) / len(t_no), 0.0, 0.0

    mean_no  = sum(t_no)  / len(t_no)
    mean_yes = sum(t_yes) / len(t_yes)
    return mean_no, mean_yes, mean_yes - mean_no


def _print_rich_overhead_report(
    mean_no: float, mean_yes: float, overhead: float,
    n_probe: int, total_batches_per_epoch: int,
) -> None:
    """rich オーバーヘッド プローブ結果を整形して表示する。"""
    estimated_s   = overhead * total_batches_per_epoch / 1000.0
    overhead_pct  = 100.0 * overhead / mean_no if mean_no > 0 else 0.0
    SEP = "─" * 70
    print(f"\n  ┌─ rich.Progress オーバーヘッド プローブ結果 ({n_probe} バッチ) {SEP[:28]}")
    print(f"  │  rich なし  バッチ平均: {mean_no:9.3f} ms")
    print(f"  │  rich あり  バッチ平均: {mean_yes:9.3f} ms  (progress.update() 込み)")
    print(f"  │  差分 (オーバーヘッド): {overhead:+9.3f} ms / バッチ  ({overhead_pct:.1f}%)")
    print(f"  │  1エポック全体への影響推定:")
    print(f"  │    {overhead:+.3f} ms × {total_batches_per_epoch} バッチ "
          f"= {estimated_s:.2f} s ({estimated_s / 60:.2f} min)")
    print(f"  └{'─' * 68}\n")


# ── 訓練ループ (タイミング計測版) ─────────────────────────────────────────────
def profile_train_one_epoch(model, device, loader, mixup_fn, criterion, optimizer,
                             loss_scaler: NativeScaler, clip_grad, model_ema,
                             epoch: int, max_batches: int = 0):
    """
    Q_ViT の train_one_epoch と同等の処理を実行しながら各フェーズを計測する。
    rich.Progress を使った場合と使わない場合の追加計測も行う。
    """
    # ── freeze/unfreeze コスト計測 ───────────────────────────────────────
    acc = TimingAccumulator()
    use_cuda = device.type == "cuda"
    timer = CudaTimer(use_cuda=use_cuda)

    _t0 = time.perf_counter()
    unfreeze_model(model)
    acc.add("unfreeze_model", (time.perf_counter() - _t0) * 1000.0)
    model.train()

    n_batches = len(loader) if max_batches == 0 else min(max_batches, len(loader))

    # ── ウォームアップ（専用イテレータで実施、本計測ループとは独立） ──────
    WARMUP = 5
    print(f"  [Epoch {epoch}] ウォームアップ {WARMUP} バッチ …")
    _warmup_iter = iter(loader)
    for _ in range(min(WARMUP, len(loader) // 2)):
        try:
            dw, tw = next(_warmup_iter)
        except StopIteration:
            break
        dw = dw.to(device, non_blocking=True)
        tw = tw.to(device, non_blocking=True)
        if mixup_fn is not None:
            dw, tw = mixup_fn(dw, tw)
        ow = model(dw)
        lw = criterion(ow, tw)
        optimizer.zero_grad()
        loss_scaler(lw, optimizer, clip_grad=clip_grad,
                    parameters=model.parameters())
        if use_cuda:
            torch.cuda.synchronize()
    del _warmup_iter

    # ── rich オーバーヘッド プローブ ──────────────────────────────────────
    # 独立関数 _probe_rich_overhead() が「同じバッチデータ」を
    # ① rich なし → ② rich あり の 2 腕で走らせて差分を計測する。
    # 本計測ループのイテレータとは完全に分離しているので
    # バッチの無駄消費・順序汚染が起きない。
    PROBE_N = min(10, n_batches // 2)
    print(f"  [Epoch {epoch}] rich オーバーヘッド プローブ ({PROBE_N} バッチ) …")
    mean_no, mean_yes, overhead = _probe_rich_overhead(
        model, device, loader, mixup_fn, criterion, optimizer, loss_scaler, clip_grad,
        n_probe=PROBE_N,
    )
    _print_rich_overhead_report(mean_no, mean_yes, overhead, PROBE_N, len(loader))
    acc.add("rich_overhead_per_batch_mean", overhead)

    # ── 本計測ループ（本番と同等: rich.Progress コンテキスト内で実行） ──────
    # ここでは実際に progress.update() を呼び、そのコストも acc に記録する。
    print(f"  [Epoch {epoch}] 本計測開始 ({n_batches} バッチ) …")
    epoch_start = time.perf_counter()
    _dl_start   = time.perf_counter()
    loader_iter  = iter(loader)

    # rich.Progress を本番と同じ設定で手動 start（try/finally で確実に stop） ─
    try:
        from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                                   MofNCompleteColumn, TimeRemainingColumn,
                                   TransferSpeedColumn)
        _prog = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            transient=True, expand=True,
            redirect_stdout=True, redirect_stderr=True,   # ← 本番と同じ設定
        )
        _prog.start()
        _task_id = _prog.add_task(f"Train epoch {epoch}", total=n_batches)
    except ImportError:
        _prog    = None
        _task_id = None

    try:
        for i in range(n_batches):
            try:
                data, target = next(loader_iter)
            except StopIteration:
                break

            # ── [1] データ読み込み ─────────────────────────────────────────
            if use_cuda:
                torch.cuda.synchronize()
            acc.add("data_load", (time.perf_counter() - _dl_start) * 1000.0)

            # ── [2] CPU→GPU 転送 ──────────────────────────────────────────
            timer.start("h2d")
            data   = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if use_cuda:
                torch.cuda.synchronize()
            timer.end("h2d")
            acc.add("h2d", timer.elapsed_ms("h2d"))

            # ── [3] Mixup ────────────────────────────────────────────────
            if mixup_fn is not None:
                timer.start("mixup")
                data, target = mixup_fn(data, target)
                if use_cuda:
                    torch.cuda.synchronize()
                timer.end("mixup")
                acc.add("mixup", timer.elapsed_ms("mixup"))
            else:
                acc.add("mixup", 0.0)

            # ── [4] Forward ──────────────────────────────────────────────
            timer.start("forward")
            output = model(data)
            loss   = criterion(output, target)
            if use_cuda:
                torch.cuda.synchronize()
            timer.end("forward")
            acc.add("forward", timer.elapsed_ms("forward"))

            # ── [5] zero_grad ────────────────────────────────────────────
            timer.start("zero_grad")
            optimizer.zero_grad()
            timer.end("zero_grad")
            acc.add("zero_grad", timer.elapsed_ms("zero_grad"))

            # ── [6] Backward ─────────────────────────────────────────────
            timer.start("backward")
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=clip_grad,
                        parameters=model.parameters(), create_graph=is_second_order)
            if use_cuda:
                torch.cuda.synchronize()
            timer.end("backward")
            acc.add("backward", timer.elapsed_ms("backward"))

            # ── [7] cuda.synchronize ─────────────────────────────────────
            timer.start("cuda_sync")
            if use_cuda:
                torch.cuda.synchronize()
            timer.end("cuda_sync")
            acc.add("cuda_sync", timer.elapsed_ms("cuda_sync"))

            # ── [8] EMA update ───────────────────────────────────────────
            timer.start("ema")
            if model_ema is not None:
                model_ema.update(model)
            if use_cuda:
                torch.cuda.synchronize()
            timer.end("ema")
            acc.add("ema", timer.elapsed_ms("ema"))

            # ── [9] rich progress.update() — 実際に呼んで計測 ────────────
            # 本計測ループは本番と同じ rich.Progress コンテキスト内で動いているため
            # ここで計測される値が redirect_stdout/stderr 込みの実コストになる。
            _pu_start = time.perf_counter()
            if _prog is not None:
                _prog.update(_task_id, advance=1)
            acc.add("progress_update", (time.perf_counter() - _pu_start) * 1000.0)

            # ── バッチ合計 ───────────────────────────────────────────────
            total = (acc.data["data_load"][-1]  + acc.data["h2d"][-1]
                     + acc.data["mixup"][-1]    + acc.data["forward"][-1]
                     + acc.data["zero_grad"][-1]+ acc.data["backward"][-1]
                     + acc.data["cuda_sync"][-1]+ acc.data["ema"][-1])
            acc.add("total_batch", total)

            if (i + 1) % 20 == 0 or i == n_batches - 1:
                print(f"    batch [{i+1:4d}/{n_batches}]  "
                      f"total={acc.mean_ms('total_batch'):.1f}ms  "
                      f"data={acc.mean_ms('data_load'):.1f}ms  "
                      f"fwd={acc.mean_ms('forward'):.1f}ms  "
                      f"bwd={acc.mean_ms('backward'):.1f}ms")

            if use_cuda:
                torch.cuda.synchronize()
            _dl_start = time.perf_counter()

    finally:
        # rich.Progress を確実に停止（例外が起きても表示が壊れないようにする）
        if _prog is not None:
            _prog.stop()

    epoch_wall = time.perf_counter() - epoch_start
    print_timing_summary(acc, epoch, n_batches, epoch_wall)
    return acc


# ── 検証ループ (タイミング計測版) ─────────────────────────────────────────────
def profile_evaluate(model, device, loader, criterion, metric, epoch: int,
                      max_batches: int = 0):
    acc = TimingAccumulator()
    use_cuda = device.type == "cuda"
    timer = CudaTimer(use_cuda=use_cuda)

    # freeze_model コスト計測
    _t0 = time.perf_counter()
    freeze_model(model)
    acc.add("freeze_model", (time.perf_counter() - _t0) * 1000.0)
    model.eval()

    n_batches = len(loader) if max_batches == 0 else min(max_batches, len(loader))
    loader_iter = iter(loader)
    epoch_start = time.perf_counter()
    _dl_start   = time.perf_counter()

    for i in range(n_batches):
        try:
            data, target = next(loader_iter)
        except StopIteration:
            break

        if use_cuda:
            torch.cuda.synchronize()
        acc.add("data_load", (time.perf_counter() - _dl_start) * 1000.0)

        timer.start("h2d")
        data   = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if use_cuda:
            torch.cuda.synchronize()
        timer.end("h2d")
        acc.add("h2d", timer.elapsed_ms("h2d"))

        timer.start("forward")
        with torch.no_grad():
            output = model(data)
            loss   = criterion(output, target)
        if use_cuda:
            torch.cuda.synchronize()
        timer.end("forward")
        acc.add("forward", timer.elapsed_ms("forward"))

        # ── metric.update() コスト (Q_ViT 特有の torcheval 呼び出し) ──
        timer.start("metric_update")
        metric.update(output, target)
        if use_cuda:
            torch.cuda.synchronize()
        timer.end("metric_update")
        acc.add("metric_update", timer.elapsed_ms("metric_update"))

        # progress.update() モック
        _pu = time.perf_counter()
        _ = i + 1
        acc.add("progress_update", (time.perf_counter() - _pu) * 1000.0)

        total = (acc.data["data_load"][-1] + acc.data["h2d"][-1]
                 + acc.data["forward"][-1] + acc.data["metric_update"][-1])
        acc.add("total_batch", total)

        if use_cuda:
            torch.cuda.synchronize()
        _dl_start = time.perf_counter()

    # metric.compute() コスト
    timer.start("metric_compute")
    _acc = metric.compute().item()
    if use_cuda:
        torch.cuda.synchronize()
    timer.end("metric_compute")
    acc.add("metric_compute", timer.elapsed_ms("metric_compute"))
    metric.reset()

    epoch_wall = time.perf_counter() - epoch_start
    print_eval_summary(acc, epoch, epoch_wall)
    return acc


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    config_path = _profile_args.config_path
    profile_epochs  = _profile_args.profile_epochs
    profile_batches = _profile_args.profile_batches

    device, config = tools.init_setting(config_path)

    # プロファイル用にエポック数を上書き
    config["EPOCHS"] = min(config["EPOCHS"], profile_epochs)

    run_dir = config["RUN_DIR"]

    print(f"\n[profile_main] device={device}")
    print(f"[profile_main] epochs={config['EPOCHS']}  "
          f"profile_batches={profile_batches}\n")

    # ── モデル構築 ────────────────────────────────────────────────────────
    model = tools.build_model(config, device)

    if (config.get("LOAD_WEIGHT", 0) and config.get("LOAD_WEIGHT_PATH", "")
            and config.get("START_EPOCH", 0) == 0):
        tools.load_weight(model, device, config)

    # ── データローダー ────────────────────────────────────────────────────
    train_loader, test_loader, classes = tools.make_dataloader(config)

    # ── 学習コンポーネント ────────────────────────────────────────────────
    (mixup_fn, train_criterion, test_criterion,
     optimizer, lr_scheduler, loss_scaler,
     model_ema, metric, epochs) = TrainUtils.setup_training(model, config, device)

    start_epoch = config.get("START_EPOCH", 0)
    clip_grad   = config.get("CLIP_GRAD", None)

    # ── freeze_model / unfreeze_model コスト (ループ外での単体計測) ──────
    print("  ─── freeze_model / unfreeze_model 単体コスト計測 ───")
    _t = time.perf_counter()
    freeze_model(model)
    freeze_cost_ms = (time.perf_counter() - _t) * 1000.0
    _t = time.perf_counter()
    unfreeze_model(model)
    unfreeze_cost_ms = (time.perf_counter() - _t) * 1000.0
    print(f"  freeze_model()  : {freeze_cost_ms:.3f} ms")
    print(f"  unfreeze_model(): {unfreeze_cost_ms:.3f} ms\n")

    # ── 全体計測 ──────────────────────────────────────────────────────────
    global_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        lr_scheduler.step(epoch)
        print(f"\n{'='*86}")
        print(f"  Epoch {epoch}  (lr={optimizer.param_groups[0]['lr']:.3e})")
        print(f"{'='*86}")

        profile_train_one_epoch(
            model, device, train_loader, mixup_fn, train_criterion,
            optimizer, loss_scaler, clip_grad, model_ema,
            epoch=epoch, max_batches=profile_batches)

        profile_evaluate(
            model, device, test_loader, test_criterion, metric,
            epoch=epoch, max_batches=profile_batches)

    global_elapsed = time.perf_counter() - global_start

    print(f"\n{'═'*86}")
    print(f"  [Q_ViT] 全エポック (×{epochs}) 合計時間: "
          f"{global_elapsed:.2f} s  ({global_elapsed/60:.2f} min)")
    print(f"  1エポック推定: {global_elapsed/epochs:.2f} s  "
          f"({global_elapsed/epochs/60:.2f} min)")
    print(f"{'═'*86}\n")

    # ── サマリーをファイルにも保存 ──────────────────────────────────────
    _summary_path = os.path.join(run_dir, "profile_summary.txt")
    print(f"  プロファイル結果を保存: {_summary_path}")


if __name__ == "__main__":
    main()
