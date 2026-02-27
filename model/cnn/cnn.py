"""
cnn.py
診断用の超軽量 CNN モデル。
ViT/Swin との比較で「遅さの原因がモデルかデータローダーか」を切り分けるために使う。

モデル構成 (tiny_cnn):
  Conv(3→32, 3x3) → BN → ReLU → MaxPool(2)   # 224→112
  Conv(32→64, 3x3) → BN → ReLU → MaxPool(2)  # 112→56
  Conv(64→128, 3x3) → BN → ReLU → MaxPool(2) # 56→28
  Conv(128→256, 3x3) → BN → ReLU → AdaptiveAvgPool(1) → Flatten
  Linear(256 → num_classes)
"""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 100, in_chans: int = 3, drop_rate: float = 0.0, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_chans, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.head(x)


def tiny_cnn(pretrained: bool = False, num_classes: int = 100,
             in_chans: int = 3, drop_rate: float = 0.0,
             drop_path_rate: float = 0.0, **kwargs) -> TinyCNN:
    """診断用超軽量 CNN（~0.35M パラメータ）"""
    return TinyCNN(num_classes=num_classes, in_chans=in_chans, drop_rate=drop_rate)
