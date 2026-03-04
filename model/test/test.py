
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
    def __init__(self, num_classes: int = 10,
                 weight_bit: int = 8, act_bit: int = 8,
                 img_size: int = 224):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        # 3 → mid=8 → 16, stride=2×2 で H/4 に縮小（224 → 56）
        # QStem 内部で qact_in / conv1+qact1 / conv2+qact2 をそれぞれ保持
        self.stem = QStem(in_chs=3, out_chs=16, weight_bit=weight_bit, act_bit=act_bit)

        # ── Stage 0: FFN ブロック (dim=16, res=56) ────────────────────────────
        # QMlp(fc1→DW→fc2) + skip 残差を QuantAct で融合する QFFN
        self.ffn1 = QFFN(dim=16, weight_bit=weight_bit, act_bit=act_bit)

        # ── Embedding 0: 16 → 32, 56 → 28（asub=False; Conv+BN fold）─────────
        # Conv3×3(s=2) + BN fold でダウンサンプル & チャネル変換
        self.emb1 = QEmbedding(in_chs=16, out_chs=32, weight_bit=weight_bit, act_bit=16)

        # ── Stage 1: FFN ブロック (dim=32, res=28) ────────────────────────────
        self.ffn2 = QFFN(dim=32, weight_bit=weight_bit, act_bit=act_bit)

        # ── Embedding 1: 32 → 48, 28 → 14（asub=False）───────────────────────
        self.emb2 = QEmbedding(in_chs=32, out_chs=48, weight_bit=weight_bit, act_bit=16)

        # ── Stage 2: AttnFFN（dim=48, res=14, stride=2 内部ダウンサンプル）──────
        # QAttention4D(stride=2) が内部で解像度を落としてからアップサンプルして戻す
        self.attn4d_s = QAttnFFN(dim=48, resolution=14, stride=2, weight_bit=weight_bit, act_bit=act_bit)

        # ── Embedding 2: 48 → 64, 14 → 7（asub=True; Attention + Conv 並列）──
        # QAttention4DDownsample（Q はストライド付き LGQuery）と
        # Conv（QEmbedding 相当）を並列で走らせ加算する
        self.attn4d_ds = QEmbeddingAttn(in_chs=48, out_chs=64, resolution=14, weight_bit=weight_bit, act_bit=act_bit)

        # ── Stage 3: AttnFFN（dim=64, res=7）─────────────────────────────────
        # stride なし: 解像度7×7 で自己注意
        self.attn4d = QAttnFFN(dim=64, resolution=7, weight_bit=weight_bit, act_bit=act_bit)

        # ── Head ──────────────────────────────────────────────────────────────
        # 最終 BN（通常の float BN; GlobalAvgPool 前に正規化）
        self.norm = nn.BatchNorm2d(64)
        # GlobalAvgPool 後の int8 テンソルを分類ヘッドへ流す量子化
        self.qact_head = QuantAct(activation_bit=act_bit)
        # 全結合分類ヘッド（QuantLinear; per-channel 重み量子化）
        self.head = QuantLinear(in_features=64, out_features=num_classes, bias=True, weight_bit=weight_bit, per_channel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # ── Stem ──────────────────────────────────────────────────────────────
        # float 入力を int8 に量子化し、Conv×2(s=2) で空間を 1/4 に縮小
        # (B, 3, 224, 224) → (B, 16, 56, 56), sf: スカラー
        x, sf = self.stem(x)

        # ── FFN 1（dim=16, res=56）────────────────────────────────────────────
        # QMlp(fc1→DW-Conv→fc2) + skip 残差（QuantAct identity で int8 に融合）
        # (B, 16, 56, 56) → (B, 16, 56, 56)
        x, sf = self.ffn1(x, sf)

        # ── Embedding 0（16→32, 56→28）────────────────────────────────────────
        # Conv3×3(s=2) + BN fold でダウンサンプル & チャネル変換 → int8 再量子化
        # (B, 16, 56, 56) → (B, 32, 28, 28)
        x, sf = self.emb1(x, sf)

        # ── FFN 2（dim=32, res=28）────────────────────────────────────────────
        # (B, 32, 28, 28) → (B, 32, 28, 28)
        x, sf = self.ffn2(x, sf)

        # ── Embedding 1（32→48, 28→14）────────────────────────────────────────
        # (B, 32, 28, 28) → (B, 48, 14, 14)
        x, sf = self.emb2(x, sf)

        # ── AttnFFN（dim=48, res=14, stride=2）────────────────────────────────
        # QAttention4D: stride=2 で内部 7×7 に縮小してアテンションを計算後 14×14 に戻す
        # QMlp: 残差 MLP で特徴を変換
        # (B, 48, 14, 14) → (B, 48, 14, 14)
        x, sf = self.attn4d_s(x, sf)

        # ── Embedding 2（48→64, 14→7; asub=True）─────────────────────────────
        # attention ストリーム（Q: LGQuery, K/V: 元解像度）と conv ストリームを並列実行し加算
        # (B, 48, 14, 14) → (B, 64, 7, 7)
        x, sf = self.attn4d_ds(x, sf)

        # ── AttnFFN（dim=64, res=7）───────────────────────────────────────────
        # 7×7 解像度で 4 ヘッド自己注意 + MLP 残差
        # (B, 64, 7, 7) → (B, 64, 7, 7)
        x, sf = self.attn4d(x, sf)

        # ── 出力前処理 ────────────────────────────────────────────────────────
        # float BN（最終正規化; GlobalAvgPool の前に適用）
        # sf は変化しない（BN は学習済みスケール/バイアスを持つ float 演算）
        x = self.norm(x)
        # GlobalAveragePooling: 空間次元 (7×7) を平均して (B, 64) に畳む
        # QuantAct の identity 機能ではなく EMA ベースで単純に再量子化する
        x = x.mean(dim=[2, 3])  # (B, 64)
        # mean pooling 後を int8 へ量子化（EMA でスケールを更新）
        x, sf = self.qact_head(x)

        # ── 分類ヘッド（QuantLinear）─────────────────────────────────────────
        # per-channel 重み量子化 + バイアス量子化 → (B, num_classes)
        # QuantLinear は内部で x / sf を行い整数行列積を再現する
        x, sf = self.head(x, sf)

        return x

#     分類ヘッドに QuantLinear を使う。

#     各レイヤーは (出力テンソル, スケーリングファクタ) のタプルを返す。
#     スケーリングファクタは次レイヤーへ渡され、量子化フローを形成する。

#     Parameters
#     ----------
#     num_classes : int
#         出力クラス数（デフォルト: 10）。
#     weight_bit : int
#         重み量子化ビット幅（デフォルト: 8）。
#     act_bit : int
#         活性化量子化ビット幅（デフォルト: 8）。
#     """

#     def __init__(self, num_classes: int = 10, weight_bit: int = 8, act_bit: int = 8):
#         super().__init__()

#         # ── 入力量子化 ─────────────────────────────────────────────────────────
#         self.quant_input = QuantAct(activation_bit=act_bit)

#         # ── Block 1: 3 → 32 ch, 224×224 → 224×224 ──────────────────────────────
#         self.conv1 = QuantBNConv2d(
#             in_channels=3, out_channels=32,
#             kernel_size=3, padding=1,
#             weight_bit=weight_bit,
#         )
#         self.relu1 = nn.ReLU6(inplace=True)
#         self.qact1 = QuantAct(activation_bit=act_bit)

#         # ── Block 2: 32 → 64 ch, 224×224 → 112×112 ─────────────────────────────
#         self.conv2 = QuantBNConv2d(
#             in_channels=32, out_channels=64,
#             kernel_size=3, stride=2, padding=1,
#             weight_bit=weight_bit,
#         )
#         self.relu2 = nn.ReLU6(inplace=True)
#         self.qact2 = QuantAct(activation_bit=act_bit)

#         # ── Block 3: 64 → 128 ch, 112×112 → 56×56 ──────────────────────────────
#         self.conv3 = QuantBNConv2d(
#             in_channels=64, out_channels=128,
#             kernel_size=3, stride=2, padding=1,
#             weight_bit=weight_bit,
#         )
#         self.relu3 = nn.ReLU6(inplace=True)
#         self.qact3 = QuantAct(activation_bit=act_bit)
        
#         # ── Block 4: 128 → 256 ch, 56×56 → 28×28 ──────────────────────────────
#         self.conv4 = QuantBNConv2d(
#             in_channels=128, out_channels=256,
#             kernel_size=3, stride=2, padding=1,
#             weight_bit=weight_bit,
#         )
#         self.relu4 = nn.ReLU6(inplace=True)
#         self.qact4 = QuantAct(activation_bit=act_bit)

#         # ── Block 5: 256 → 512 ch, 28×28 → 14×14 ──────────────────────────────
#         self.conv5 = QuantBNConv2d(
#             in_channels=256, out_channels=512,
#             kernel_size=3, stride=2, padding=1,
#             weight_bit=weight_bit,
#         )
#         self.relu5 = nn.ReLU6(inplace=True)
#         self.qact5 = QuantAct(activation_bit=act_bit)

#         # ──  分類ヘッド ───────────────────────────────
#         self.fc = QuantLinear(
#             in_features=512*14*14, out_features=num_classes,
#             bias=True,
#             weight_bit=weight_bit,
#             per_channel=True,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         x : torch.Tensor
#             入力テンソル。shape = (B, 3, H, W)。

#         Returns
#         -------
#         torch.Tensor
#             クラスロジット。shape = (B, num_classes)。
#         """
#         # ── 入力量子化 ─────────────────────────────────────────────────────────
#         # pre_act_scaling_factor=None → SymmetricQuantFunction で直接量子化
#         x, sf = self.quant_input(x)

#         # ── Block 1 ────────────────────────────────────────────────────────────
#         x, sf = self.conv1(x, sf)
#         x = self.relu1(x)
#         x, sf = self.qact1(x, sf)

#         # ── Block 2 ────────────────────────────────────────────────────────────
#         x, sf = self.conv2(x, sf)
#         x = self.relu2(x)
#         x, sf = self.qact2(x, sf)

#         # ── Block 3 ────────────────────────────────────────────────────────────
#         x, sf = self.conv3(x, sf)
#         x = self.relu3(x)
#         x, sf = self.qact3(x, sf)

#         # ── Block 4 ────────────────────────────────────────────────────────────
#         x, sf = self.conv4(x, sf)
#         x = self.relu4(x)
#         x, sf = self.qact4(x, sf)
        
#         # ── Block 5 ────────────────────────────────────────────────────────────
#         x, sf = self.conv5(x, sf)
#         x = self.relu5(x)
#         x, sf = self.qact5(x, sf)
        
#         # ── (B, 128, 1, 1) → (B, 128) ───────────────
#         x = torch.flatten(x, 1)

#         # ── 全結合 (量子化) ────────────────────────────────────────────────────
#         # QuantLinear は内部で x / sf を行い整数演算を再現する
#         x, sf = self.fc(x, sf)

#         return x


# class TestCNN(nn.Module):
#     def __init__(self, num_classes: int = 10, weight_bit: int = 8, act_bit: int = 8):
#         super().__init__()

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU6(inplace=True)

#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.relu2 = nn.ReLU6(inplace=True)

#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.relu3 = nn.ReLU6(inplace=True)
        
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.relu4 = nn.ReLU6(inplace=True)

#         self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.relu5 = nn.ReLU6(inplace=True)

#         self.fc = nn.Linear(512 * 14 * 14, num_classes, bias=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)

#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu4(x)
        
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
        
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

