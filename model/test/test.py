"""
test.py
QuantBNConv2d / nn.ReLU6 / QuantAct / QuantLinear を用いた簡単な量子化 CNN モデル。

ネットワーク構造（CIFAR-10 を想定: 入力 3×32×32）:
    入力 → QuantAct
    Block1: QuantBNConv2d(3→32, 3×3, pad=1) → ReLU6 → QuantAct
    Block2: QuantBNConv2d(32→64, 3×3, stride=2, pad=1) → ReLU6 → QuantAct
    Block3: QuantBNConv2d(64→128, 3×3, stride=2, pad=1) → ReLU6 → QuantAct
    AdaptiveAvgPool2d(1) → Flatten
    QuantLinear(128 → num_classes)
"""

import torch
import torch.nn as nn

from .module import QuantAct, QuantBNConv2d, QuantLinear, QuantConv2d


def test_cnn(pretrained: bool = False,
             num_classes: int = 10,
             in_chans: int = 3,
             drop_rate: float = 0.0,
             drop_path_rate: float = 0.0,
             weight_bit: int = 8,
             act_bit: int = 8,
             **kwargs) -> 'TestCNN':
    """
    utils.build_model から呼ばれるファクトリ関数。
    pretrained / in_chans / drop_rate / drop_path_rate は
    インターフェース統一のため受け取るが TestCNN では未使用。
    """
    model = TestCNN(num_classes=num_classes, weight_bit=weight_bit, act_bit=act_bit)
    return model


class TestCNN(nn.Module):
    """
    量子化 CNN の簡易実装。

    すべての畳み込みに BN 折り畳み付きの QuantBNConv2d、
    活性化に nn.ReLU6 と QuantAct、
    分類ヘッドに QuantLinear を使う。

    各レイヤーは (出力テンソル, スケーリングファクタ) のタプルを返す。
    スケーリングファクタは次レイヤーへ渡され、量子化フローを形成する。

    Parameters
    ----------
    num_classes : int
        出力クラス数（デフォルト: 10）。
    weight_bit : int
        重み量子化ビット幅（デフォルト: 8）。
    act_bit : int
        活性化量子化ビット幅（デフォルト: 8）。
    """

    def __init__(self, num_classes: int = 10, weight_bit: int = 8, act_bit: int = 8):
        super().__init__()

        # ── 入力量子化 ─────────────────────────────────────────────────────────
        self.quant_input = QuantAct(activation_bit=act_bit)

        # ── Block 1: 3 → 32 ch, 224×224 → 224×224 ──────────────────────────────
        self.conv1 = QuantBNConv2d(
            in_channels=3, out_channels=32,
            kernel_size=3, padding=1,
            weight_bit=weight_bit,
        )
        self.relu1 = nn.ReLU6(inplace=True)
        self.qact1 = QuantAct(activation_bit=act_bit)

        # ── Block 2: 32 → 64 ch, 224×224 → 112×112 ─────────────────────────────
        self.conv2 = QuantBNConv2d(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=2, padding=1,
            weight_bit=weight_bit,
        )
        self.relu2 = nn.ReLU6(inplace=True)
        self.qact2 = QuantAct(activation_bit=act_bit)

        # ── Block 3: 64 → 128 ch, 112×112 → 56×56 ──────────────────────────────
        self.conv3 = QuantBNConv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=2, padding=1,
            weight_bit=weight_bit,
        )
        self.relu3 = nn.ReLU6(inplace=True)
        self.qact3 = QuantAct(activation_bit=act_bit)
        
        # ── Block 4: 128 → 256 ch, 56×56 → 28×28 ──────────────────────────────
        self.conv4 = QuantBNConv2d(
            in_channels=128, out_channels=256,
            kernel_size=3, stride=2, padding=1,
            weight_bit=weight_bit,
        )
        self.relu4 = nn.ReLU6(inplace=True)
        self.qact4 = QuantAct(activation_bit=act_bit)

        # ── Block 5: 256 → 512 ch, 28×28 → 14×14 ──────────────────────────────
        self.conv5 = QuantBNConv2d(
            in_channels=256, out_channels=512,
            kernel_size=3, stride=2, padding=1,
            weight_bit=weight_bit,
        )
        self.relu5 = nn.ReLU6(inplace=True)
        self.qact5 = QuantAct(activation_bit=act_bit)

        # ──  分類ヘッド ───────────────────────────────
        self.fc = QuantLinear(
            in_features=512*14*14, out_features=num_classes,
            bias=True,
            weight_bit=weight_bit,
            per_channel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力テンソル。shape = (B, 3, H, W)。

        Returns
        -------
        torch.Tensor
            クラスロジット。shape = (B, num_classes)。
        """
        # ── 入力量子化 ─────────────────────────────────────────────────────────
        # pre_act_scaling_factor=None → SymmetricQuantFunction で直接量子化
        x, sf = self.quant_input(x)

        # ── Block 1 ────────────────────────────────────────────────────────────
        x, sf = self.conv1(x, sf)
        x = self.relu1(x)
        x, sf = self.qact1(x, sf)

        # ── Block 2 ────────────────────────────────────────────────────────────
        x, sf = self.conv2(x, sf)
        x = self.relu2(x)
        x, sf = self.qact2(x, sf)

        # ── Block 3 ────────────────────────────────────────────────────────────
        x, sf = self.conv3(x, sf)
        x = self.relu3(x)
        x, sf = self.qact3(x, sf)

        # ── Block 4 ────────────────────────────────────────────────────────────
        x, sf = self.conv4(x, sf)
        x = self.relu4(x)
        x, sf = self.qact4(x, sf)
        
        # ── Block 5 ────────────────────────────────────────────────────────────
        x, sf = self.conv5(x, sf)
        x = self.relu5(x)
        x, sf = self.qact5(x, sf)
        
        # ── (B, 128, 1, 1) → (B, 128) ───────────────
        x = torch.flatten(x, 1)

        # ── 全結合 (量子化) ────────────────────────────────────────────────────
        # QuantLinear は内部で x / sf を行い整数演算を再現する
        x, sf = self.fc(x, sf)

        return x
