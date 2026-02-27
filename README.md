# 量子化 Vision Transformer

Integer-only arithmetic による量子化 DeiT / Swin Transformer の PyTorch 実装です。  

---

## セットアップ


### Requirements
- Python 3.12
- CUDA 13.0


### CIFAR-10 / CIFAR-100

`config.json` の `DATA_SET` を `"CIFAR10"` または `"CIFAR100"` に設定するだけで、  
初回実行時に `data/` へ自動ダウンロードされます。

### ImageNet

```bash
ln -s /path/to/imagenet test/data/imagenet
```
`config.json` の `DATA_SET` を `"IMNET"`, `DATA_DIR` をデータセットパスに設定してください。

---

## 使い方

```bash
cd test
python main.py              # config.json を自動検索
python main.py config.json  # パスを明示指定
```

---

## 参考

- [I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference (ICCV 2023)](https://arxiv.org/abs/2207.01405)
- [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
