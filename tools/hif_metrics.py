"""Numpy-based HIF evaluation metrics used across model test scripts.

Keeps dependencies consistent with the repo (sewar + scipy).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import uniform_filter
from sewar.full_ref import psnr as _psnr
from sewar.full_ref import rmse_sw, sam as _sam, ssim as _ssim
from sewar.utils import _initial_check


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / np.iinfo(arr.dtype).max).clip(0.0, 1.0)

    arr = arr.astype(np.float32)
    maxv = float(np.nanmax(arr)) if arr.size else 1.0
    minv = float(np.nanmin(arr)) if arr.size else 0.0

    # Common cases: already 0..1, or 0..255-ish floats.
    if maxv <= 1.0 and minv >= 0.0:
        return arr
    if maxv > 1.0 and minv >= 0.0:
        return (arr / maxv).clip(0.0, 1.0)

    # Fallback: shift+scale to 0..1
    denom = (maxv - minv) if (maxv - minv) != 0 else 1.0
    return ((arr - minv) / denom).clip(0.0, 1.0)


def ergas(gt: np.ndarray, pred: np.ndarray, ratio: int = 4, ws: int = 8) -> float:
    """ERGAS metric (robust to NaNs).

    Adapted from main/metrics.py (which is adapted from sewar.full_ref.ergas).
    """

    gt, pred = _initial_check(gt, pred)
    _, rmse_map = rmse_sw(gt, pred, ws)
    means_map = uniform_filter(gt, ws) / ws**2

    # Avoid division by zero
    idx = means_map == 0
    means_map[idx] = 1
    rmse_map[idx] = 0

    nb = 1
    ergasroot = np.sqrt(np.sum(((rmse_map**2) / (means_map**2)), axis=2) / nb)
    ergas_map = 100 * ratio * ergasroot

    s = int(np.round(ws / 2))
    return float(np.nanmean(ergas_map[s:-s, s:-s]))


def compute_metrics(gt: np.ndarray, pred: np.ndarray, ratio: int = 4) -> Dict[str, float]:
    gt01 = normalize01(gt)
    pr01 = normalize01(pred)

    return {
        "psnr": float(_psnr(gt01, pr01, MAX=1.0)),
        "ssim": float(_ssim(gt01, pr01, MAX=1.0)[0]),
        "sam": float(_sam(gt01, pr01)),
        "ergas": float(ergas(gt01, pr01, ratio=ratio)),
    }
