"""
utils.py (model/swin)
Swin 事前学習済み重みロードユーティリティ（NPZ 形式）
"""

import pretty_errors
import os
import math
import warnings
from itertools import repeat
import collections.abc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ── ユーティリティ ────────────────────────────────────────────────────────────
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@torch.no_grad()
def load_weights_from_npz(model, url, check_hash=False, progress=False, prefix=''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation """

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    def _get_cache_dir(child_dir=''):
        hub_dir = torch.hub.get_dir()
        child_dir = () if not child_dir else (child_dir,)
        model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _download_cached_file(url, check_hash=True, progress=False):
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(_get_cache_dir(), filename)
        if not os.path.exists(cached_file):
            hash_prefix = None
            if check_hash:
                r = torch.hub.HASH_REGEX.search(filename)
                hash_prefix = r.group(1) if r else None
            torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=progress)
        return cached_file

    cached_file = _download_cached_file(url, check_hash=check_hash, progress=progress)
    w = np.load(cached_file)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'
