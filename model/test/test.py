
import torch
import torch.nn as nn

from .module import (
    QuantAct, QuantBNConv2d, QuantLinear, QuantConv2d,
    DropPath, QStem, QMlp, QFFN,
    QAttention4D, QAttnFFN, QEmbedding, QEmbeddingAttn,
)


def test_cnn(pretrained: bool = False,
             num_classes: int = 10,
             in_chans: int = 3,
             drop_rate: float = 0.0,
             drop_path_rate: float = 0.0,
             weight_bit: int = 8,
             act_bit: int = 8,
             **kwargs) -> 'test':

    model = test(num_classes=num_classes, weight_bit=weight_bit, act_bit=act_bit)
    return model


class test(nn.Module):
    def __init__(self, num_classes: int = 10, weight_bit: int = 8, act_bit: int = 8, img_size: int = 224):
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:


        return x

