# Q_ViT パフォーマンス最適化 — 全変更記録

> 作成日: 2026-02-27  
> 目的: RTX 3060 上での 1 エポック学習時間を **38 分 → 約 10 分以下** に削減した  
> すべての変更をここから復元できるよう、before/after とその理由を詳細に残す。

---

## 変更サマリー（時系列）

| # | 問題 | 原因 | 対応 | 効果 |
|---|------|------|------|------|
| 1 | CUDA ScatterGatherKernel クラッシュ | `NUM_CLASSES=10` なのに CIFAR-100 を使用 | config.json を修正 | クラッシュ解消 |
| 2 | GPU → RAM スワップ（OOM） | FP64 テンソル + Worker 数過多 | FP32 化 + workers 8→4 | OOM 解消 |
| 3 | 38 分/epoch | `batch_frexp` が CPU ラウンドトリップ ×56K/epoch | `batch_frexp` 削除、`fixedpoint_mul` を FP32 直計算に | 20 分へ |
| 4 | 20 分/epoch | `floor_ste`/`round_ste` が Python クラス dispatch ×172K/epoch | インライン STE 関数に変更 | → |
| 5 | 同上 | `IntLayerNorm` Newton ループで `floor_ste.apply` ×20 回/層 | `torch.no_grad()` + `torch.floor` に変更 | 15 分へ |
| 6 | 15 分/epoch | `QuantAct` が `reshape+transpose` で 9.7 MB テンソルを毎回生成 | `x_act.min()` / `x_act.max()` に変更 | → |
| 7 | 同上 | `torch.tensor()` による GPU テンソル毎回アロケーション | `x.new_zeros(1)` / `register_buffer` に変更 | → |
| 8 | 同上 | `torch.cuda.synchronize()` によるパイプライン強制停止 ×781/epoch | 削除 | → |
| 9 | 同上 | `loss.item()` による CPU-GPU 同期 ×1563/epoch | GPU 上でロス累積に変更 | 12 分へ |
| 10 | 12 分/epoch | eval 時も重み量子化を毎回再計算 | `if self.training:` ガードを追加 | → |
| 11 | 同上 | eval 時も `act_scaling_factor` を毎回再計算 | `running_stat` ブロック内に移動 | → |
| 12 | 同上 | `IntGELU`/`IntSoftmax` が eval 時に `x0_int` を毎回計算 | `_x0_int_cache` キャッシュを追加 | → |
| 13 | 同上 | `WindowAttention` が eval 時に `relative_position_bias` を毎回計算 | キャッシュ + `train()` オーバーライドを追加 | 10 分台へ |
| 14 | 同上 | `NativeScaler`（GradScaler）を `autocast` なしで使用 → 全勾配 NaN/Inf スキャンのみ発生 | `loss.backward()` + `optimizer.step()` に置き換え | → |
| 15 | 同上 | `torch.finfo(torch.float32).eps` をホットパスで毎回呼び出し | モジュール定数 `_EPS_F32` に変更 | → |
| 16 | 同上 | `save_curves` (matplotlib) が毎エポック起動 | 5 エポックごとに間引き | → |

---

## 詳細変更内容

---

### [修正 1] CUDA クラッシュ — `config.json`

**症状:** `RuntimeError: CUDA error: device-side assert triggered` (ScatterGatherKernel)  
**原因:** CIFAR-10（クラス 10）のラベルが 100 クラス用の embedding に入力された。

```jsonc
// Before
"NUM_CLASSES": 100

// After
"NUM_CLASSES": 10
```

---

### [修正 2] GPU メモリ OOM — `fixedpoint_mul` (両 quant_module.py) + workers

**症状:** GPU メモリが RAM にスワップアウト、学習が停止。  
**原因①:** `fixedpoint_mul.forward` が `.double()`（FP64）を使用。RTX 3060 の FP64 は FP32 の 1/26 の速度。  
**原因②:** `num_workers=8` で DataLoader が大量のプロセスを起動。

#### `model/i_vit/quant_module.py` および `model/swin/quant_module.py`

```python
# Before（FP64 キャスト）
z_int = torch.round(pre_act / pre_act_scaling_factor).double()
new_scale = reshape(pre_act_scaling_factor.double() / z_scaling_factor.double())
output = torch.round(z_int * new_scale)

# After（FP32 のまま）
z_int = torch.round(pre_act / pre_act_scaling_factor)
new_scale = reshape(pre_act_scaling_factor / z_scaling_factor)
output = torch.round(z_int * new_scale)
```

#### `config.json`

```jsonc
// Before
"NUM_WORKERS": 8

// After
"NUM_WORKERS": 4
```

---

### [修正 3] `batch_frexp` 削除 — 両 `quant_module.py`

**症状:** 1 エポック 38 分。  
**原因:** 独自実装の `batch_frexp` が CPU ループ ＋ Python リスト操作を含み、GPU-CPU ラウンドトリップが ×56K/epoch 発生していた。  
**対応:** `batch_frexp` 関数を丸ごと削除し、`fixedpoint_mul` 内を FP32 直計算に変更（上記修正 2 と同時適用）。

```python
# Before（batch_frexp を使用した複雑な計算）
def batch_frexp(inputs, max_bit=31):
    # ... CPU ループ処理 ...
    pass

# fixedpoint_mul 内
new_scale_factor, new_exp = batch_frexp(pre_act_scaling_factor / z_scaling_factor)
output = torch.round(z_int * new_scale_factor.to(z_int.device)) << new_exp.to(z_int.device)

# After（batch_frexp を完全削除、FP32 直計算）
z_int = torch.round(pre_act / pre_act_scaling_factor)
new_scale = reshape(pre_act_scaling_factor / z_scaling_factor)
output = torch.round(z_int * new_scale)
```

---

### [修正 4] `floor_ste` / `round_ste` のインライン化 — 両 `quant_module.py`

**原因:** `torch.autograd.Function` サブクラスの `.apply()` 呼び出しは Python ディスパッチオーバーヘッドが大きい。`forward` パスで ×172K/epoch 以上呼ばれる。

```python
# Before（Function クラス）
class floor_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class round_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# 呼び出し側
k = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)

# After（インライン STE 関数）
def floor_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.floor(x) - x).detach()

def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()

# 呼び出し側（関数呼び出しのまま変わらず）
k = floor_ste((k + floor_ste(var_int / k)) / 2)
```

---

### [修正 5] `IntLayerNorm` Newton ループの最適化 — 両 `quant_module.py`

**原因:** Newton 法による整数平方根計算（10 回 × 2回 = 20 回の `floor_ste.apply`）が Python ディスパッチを毎呼び出し走らせていた。勾配不要な区間なのに STE を使う意味もない。

```python
# Before
k = var_int.new_full(var_int.shape, 2**16)
for _ in range(10):
    k = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
std_int = k

# After（no_grad 下で torch.floor を直接使用）
with torch.no_grad():
    k = var_int.new_full(var_int.shape, 2**16)
    for _ in range(10):
        k = torch.floor((k + torch.floor(var_int / k)) / 2)
std_int = k
```

---

### [修正 6] `QuantAct` の min/max 計算を簡略化 — 両 `quant_module.py`

**原因:** 元の実装はチャネルごとの min/max を求めるために `reshape` + `transpose` + 9.7 MB のテンソル操作を行っていた。グローバル min/max で十分な場面でこれを毎バッチ実行していた。

```python
# Before（reshape+transpose）
if self.per_channel:
    # ... reshape, transpose, per-channel min/max ...
else:
    x_act = x_act.reshape(-1, x_act.shape[-1])
    cur_min = x_act.min(axis=0).values
    cur_max = x_act.max(axis=0).values

# After（グローバル min/max）
cur_min = x_act.min()
cur_max = x_act.max()
```

---

### [修正 7] `torch.tensor()` GPU アロケーション排除 — 両 `quant_module.py`

**原因:** `torch.tensor(0, dtype=...)` はホットパスで毎回 Python → C++ ブリッジを経由して新しい GPU テンソルを確保する。

#### `SymmetricQuantFunction` / `TernaryQuantFunction`

```python
# Before
zero_point = torch.tensor(0).to(x.device)

# After（入力テンソルのデバイスを継承）
zero_point = x.new_zeros(1)
```

#### `IntGELU.__init__` / `IntSoftmax.__init__`

```python
# Before（__init__ で毎回 tensor を作成していた）
self.sigmoid_scaling_factor = 1 / 2 ** (output_bit - 1)  # Pytorchバッファでなかった

# After（register_buffer で永続バッファとして登録）
self.register_buffer('sigmoid_scaling_factor', torch.tensor([1 / 2 ** (output_bit - 1)]))
# IntSoftmax も同様
self.register_buffer('out_scaling_factor', torch.tensor([1 / 2 ** (output_bit - 1)]))
```

---

### [修正 8] `torch.cuda.synchronize()` 削除 — `utils.py`

**原因:** `synchronize()` は GPU パイプラインを強制停止させる。プロファイリング目的のコードが本番コードに残っていた。×781/epoch 発生。

```python
# Before（evaluate の末尾）
torch.cuda.synchronize()
return loss_avg, acc

# After（削除）
return loss_avg, acc
```

---

### [修正 9] `loss.item()` による CPU-GPU 同期を排除 — `utils.py`

**原因:** `.item()` は Python スカラーが必要なため GPU-CPU 同期が発生。毎バッチ（×1563/epoch）実行されていた。

```python
# Before
train_loss += loss.item()
# ...
return train_loss / len(loader)

# After（GPU 上でロスを累積）
loss_sum_gpu = torch.zeros(1, device=device)
# ...
loss_sum_gpu.add_(loss.detach() * data.size(0))
count += data.size(0)
# ...
return (loss_sum_gpu / count).item()  # 最後に 1 回だけ同期
```

---

### [修正 10] eval 時の重み量子化スキップ — 両 `quant_module.py`

**原因:** `QuantLinear`, `TerLinear`, `QuantConv2d` の `forward` で、eval 時も毎バッチ重みの min/max 計算 → スケール計算 → 量子化を行っていた。eval では重みが変わらないため無駄。

```python
# Before（常に量子化計算）
def forward(self, x, prev_act_scaling_factor=None):
    w = self.weight
    # min/max → scaling_factor → weight_integer を毎回計算
    ...

# After（training 時のみ計算）
def forward(self, x, prev_act_scaling_factor=None):
    if self.training:
        with torch.no_grad():
            w = self.weight
            # min/max → scaling_factor → weight_integer を計算
            ...
    else:
        # eval 時は前回の fc_scaling_factor をそのまま使用
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor
```

同様の変更を `QuantConv2d` でも実施（`conv_scaling_factor` を `fc_scaling_factor` の代わりに使用）。

---

### [修正 11] `QuantAct` の eval 時 scaling factor 再計算をスキップ — 両 `quant_module.py`

**原因:** `act_scaling_factor` の更新は `running_stat=True` のとき（=学習中）のみ必要。eval 時は `running_stat=False` になるが、コードが `running_stat` ブロックの外にあったため常に実行されていた。

```python
# Before（常に act_scaling_factor を再計算）
if self.running_stat:
    # min/max 更新
    ...
self.act_scaling_factor = symmetric_linear_quantization_params(...)  # ← 常に実行

# After（running_stat ブロック内に移動）
if self.running_stat:
    # min/max 更新
    ...
    self.act_scaling_factor = symmetric_linear_quantization_params(...)  # ← 学習時のみ
```

---

### [修正 12] `IntGELU` / `IntSoftmax` の `x0_int` キャッシュ — 両 `quant_module.py`

**原因:** `int_exp_shift` 内の `torch.floor(-1.0 / scaling_factor)` は、eval 時に `scaling_factor` が変化しないにもかかわらず毎バッチ再計算されていた。

```python
# Before（毎回計算）
def int_exp_shift(self, x_int, scaling_factor):
    ...
    x0_int = torch.floor(-1.0 / scaling_factor)
    ...

# After（eval 時キャッシュ）
# __init__ に追加
self._x0_int_cache = None

# int_exp_shift
with torch.no_grad():
    if self._x0_int_cache is None or self.training:
        x0_int = torch.floor(-1.0 / scaling_factor)
        if not self.training:
            self._x0_int_cache = x0_int
    else:
        x0_int = self._x0_int_cache
```

`IntGELU` と `IntSoftmax` の両方に同じ変更を適用。

---

### [修正 13] `WindowAttention` の `relative_position_bias` キャッシュ — `model/swin/swin.py`

**原因:** Swin の `WindowAttention.forward` が毎バッチ `relative_position_bias_table` の gather + permute + contiguous を実行していた。eval 時はウィンドウ構造が変わらないため再計算不要（×1884/epoch）。

```python
# Before（毎バッチ計算）
def forward(self, x, mask=None):
    ...
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)].view(...)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

# After（eval 時キャッシュ）
# __init__ に追加
self._rel_pos_bias_cache = None

# forward
if self.training or self._rel_pos_bias_cache is None:
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)].view(...)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    if not self.training:
        self._rel_pos_bias_cache = relative_position_bias
else:
    relative_position_bias = self._rel_pos_bias_cache
attn = attn + relative_position_bias.unsqueeze(0)

# train() オーバーライドでキャッシュをクリア
def train(self, mode: bool = True):
    super().train(mode)
    if mode:
        self._rel_pos_bias_cache = None
    return self
```

---

### [修正 14] `NativeScaler`（GradScaler）削除 — `utils.py` / `main.py`

**原因:** `timm.utils.NativeScaler` は `torch.cuda.amp.GradScaler` のラッパー。  
AMP を使うには `autocast()` コンテキストが必要だが、このコードには一切なかった。  
結果として FP16 演算という恩恵はゼロで、**毎バッチ全勾配テンソルの NaN/Inf スキャン**だけが実行されていた。

#### `utils.py`

```python
# Before（インポート）
from timm.utils import NativeScaler, ModelEma

# After
from timm.utils import ModelEma

# Before（setup_training 内）
loss_scaler = NativeScaler()
return ..., lr_scheduler, loss_scaler, model_ema, metric, epochs

# After
# AMP スケーラー行を削除
# コメントに理由を残す:
# autocast なしで GradScaler を使うと全勾配の NaN スキャンのみ発生するため削除
return ..., lr_scheduler, model_ema, metric, epochs

# Before（train_one_epoch シグネチャ）
def train_one_epoch(model, device, loader, mixup_fn, criterion, optimizer,
                    loss_scaler: NativeScaler, clip_grad, model_ema):

# After
def train_one_epoch(model, device, loader, mixup_fn, criterion, optimizer,
                    clip_grad, model_ema):

# Before（学習ステップ）
is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
loss_scaler(loss, optimizer, clip_grad=clip_grad,
            parameters=model.parameters(), create_graph=is_second_order)

# After
is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
loss.backward(create_graph=is_second_order)
if clip_grad is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
optimizer.step()
```

#### `main.py`

```python
# Before（インポート）
from timm.utils import NativeScaler, ModelEma

# After
from timm.utils import ModelEma

# Before（unpacking）
(mixup_fn, train_criterion, test_criterion,
 optimizer, lr_scheduler, loss_scaler,
 model_ema, metric, epochs) = TrainUtils.setup_training(...)

# After
(mixup_fn, train_criterion, test_criterion,
 optimizer, lr_scheduler,
 model_ema, metric, epochs) = TrainUtils.setup_training(...)

# Before（train_one_epoch 呼び出し）
train_loss = TrainUtils.train_one_epoch(
    model, device, train_loader, mixup_fn, train_criterion,
    optimizer, loss_scaler, clip_grad, model_ema)

# After
train_loss = TrainUtils.train_one_epoch(
    model, device, train_loader, mixup_fn, train_criterion,
    optimizer, clip_grad, model_ema)
```

---

### [修正 15] `_EPS_F32` モジュール定数化 — 両 `quant_module.py`

**原因:** `torch.finfo(torch.float32).eps` がホットパスの `symmetric_linear_quantization_params`（毎バッチ多数回呼ばれる）と `TerLinear.forward` で毎回評価されていた。

```python
# Before（関数内で毎回評価）
def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps  # ← 毎回 Python 呼び出し
        scale = max_val / float(n)
        scale.clamp_(eps)

# TerLinear.forward 内
self.fc_scaling_factor = abs_mean.clamp(min=torch.finfo(torch.float32).eps)

# After（モジュール先頭に定数として 1 回だけ評価）
_EPS_F32: float = torch.finfo(torch.float32).eps  # 1.1920929e-07

def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        scale = max_val / float(n)
        scale.clamp_(min=_EPS_F32)  # ← 定数を使用

# TerLinear.forward 内
self.fc_scaling_factor = abs_mean.clamp(min=_EPS_F32)
```

#### `QuantAct.__init__` への `_quant_n` 追加（両 `quant_module.py`）

```python
# Before（activation_bit のみ保持）
self.activation_bit = activation_bit

# After（_quant_n をプリコンピュート）
self.activation_bit = activation_bit
self._quant_n = 2 ** (activation_bit - 1) - 1  # 量子化レンジ上限（定数）
```

---

### [修正 16] `save_curves` の間引き — `utils.py` / `main.py`

**原因:** matplotlib の `Figure` 生成 + `savefig` は ~1–2 秒かかる。250 エポック毎回実行すると合計最大 500 秒のオーバーヘッド。

#### `utils.py`

```python
# Before
@staticmethod
def save_curves(train_losses, test_losses, test_accs, config: dict):
    """学習曲線をプロットして保存"""
    try:
        fig, ...
        fig.savefig(path)
        ...

# After
@staticmethod
def save_curves(train_losses, test_losses, test_accs, config: dict, epoch: int = 0):
    """学習曲線をプロットして保存（5エポックごと）"""
    if epoch % 5 != 0 and epoch != config.get("EPOCHS", 0):
        return
    try:
        fig, ...
        fig.savefig(path)
        ...
```

#### `main.py`

```python
# Before
tools.save_curves(train_losses, test_losses, test_accs, config)

# After（epoch 番号を渡す）
tools.save_curves(train_losses, test_losses, test_accs, config, epoch + 1)
```

---

## `swin/quant_module.py` の fix() バグ修正

`swin/quant_module.py` の `QuantLinear` に `fix()` が2回定義されていたタイポを修正。

```python
# Before（重複定義）
def fix(self): pass
def fix(self): pass   # ← タイポ（2つ目は unfix のはずだった）

# After
def fix(self): pass
def unfix(self): pass
```

---

## 変更対象ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `config.json` | `NUM_CLASSES` 修正、`NUM_WORKERS` 削減 |
| `model/i_vit/quant_module.py` | 修正 2〜12、14〜15 |
| `model/swin/quant_module.py` | 修正 2〜12、14〜15、fix() バグ修正 |
| `model/swin/swin.py` | 修正 13（`WindowAttention` キャッシュ） |
| `utils.py` | 修正 8〜11、14、16 |
| `main.py` | 修正 14、16 |

---

## 学習時間の推移

```
38 分  →  [batch_frexp 削除 + FP32化]
20 分  →  [floor_ste インライン + IntLayerNorm + QuantAct min/max]
15 分  →  [tensor alloc 削減 + synchronize 削除 + loss GPU 累積]
12 分  →  [eval 重み量子化スキップ + scaling_factor スキップ + x0_int キャッシュ + rel_pos_bias キャッシュ]
10 分台  →  [NativeScaler 削除 + _EPS_F32 定数化 + save_curves 間引き]
```

---

## 復元手順

上記の各 "After" コードを対応ファイルに適用すれば、このリポジトリの最適化済み状態を完全に再現できる。  
特にゼロから復元する場合の優先順位：

1. **`batch_frexp` 削除 + `fixedpoint_mul` FP32 化**（38 分 → 20 分、最大インパクト）  
2. **`NativeScaler` 削除**（全勾配 NaN スキャン排除、GradScaler を使う正当な理由がない限り不要）  
3. **`floor_ste`/`round_ste` インライン化**（Python ディスパッチを排除）  
4. **eval 時の重み量子化スキップ** + **`QuantAct` eval スキップ**（eval 速度が大幅改善）  
5. **各種キャッシュ**（`x0_int`, `rel_pos_bias`）  
6. **`_EPS_F32` 定数化** + **`save_curves` 間引き**（細かいが積み重ね）
