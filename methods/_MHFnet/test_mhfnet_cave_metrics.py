"""Evaluate CMHF-net outputs on the 12-image CAVE test split.

This script computes SSIM, SAM, ERGAS using the same formulas used in
methods/_DBIN/dbintest.py (bandwise SSIM averaged over bands; SAM in degrees;
ERGAS from per-band MSE and per-band mean).

Expected inputs:
- Ground truth + prepared split list in CMHF-net format:
  - CAVEdata/List (contains Ind: 1-based indices; first 20 train, last 12 test)
  - CAVEdata/X/<scene>.mat with key 'msi' (H,W,31) in [0,1]
- Predictions produced by CMHF-net testAll:
  - <pred_dir>/<scene>.mat with key 'outX' (1,H,W,31) or (H,W,31)

Example:
  python methods/_MHFnet/test_mhfnet_cave_metrics.py \
    --cmhf_root methods/_MHFnet/CMHF-net \
    --pred_dir methods/_MHFnet/CMHF-net/TestResult/Result \
    --sf 32
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.io as sio

try:
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:  # pragma: no cover
    # Older skimage compatibility
    from skimage.measure import compare_ssim  # type: ignore


def _squeeze_hw_c(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC or 1xHxWxC, got shape {arr.shape}")
    return arr


def compute_ms_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """DBIN style: average SSIM across bands."""
    image1 = _squeeze_hw_c(image1)
    image2 = _squeeze_hw_c(image2)
    n = image1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        ms_ssim += float(compare_ssim(image1[:, :, i], image2[:, :, i], data_range=1.0))
    return ms_ssim / float(n)


def compute_sam(image1: np.ndarray, image2: np.ndarray) -> float:
    """DBIN style: mean SAM over pixels (degrees)."""
    image1 = _squeeze_hw_c(image1)
    image2 = _squeeze_hw_c(image2)
    h, w, c = image1.shape
    a = np.reshape(image1, (h * w, c))
    b = np.reshape(image2, (h * w, c))
    mole = np.sum(a * b, axis=1)
    a_norm = np.sqrt(np.sum(a * a, axis=1))
    b_norm = np.sqrt(np.sum(b * b, axis=1))
    deno = a_norm * b_norm
    sam = np.rad2deg(np.arccos((mole + 1e-11) / (deno + 1e-11)))
    return float(np.mean(sam))


def compute_ergas_from_mse(mse_per_band: np.ndarray, out: np.ndarray, sf: int) -> float:
    """DBIN style ERGAS from per-band MSE and per-band mean of output."""
    out = _squeeze_hw_c(out)
    h, w, c = out.shape
    out2 = np.reshape(out, (h * w, c))
    out_mean = np.mean(out2, axis=0)
    mse_per_band = np.reshape(mse_per_band, (c, 1))
    out_mean = np.reshape(out_mean, (c, 1))
    ergas = 100.0 / float(sf) * np.sqrt(np.mean(mse_per_band / (out_mean ** 2 + 1e-12)))
    return float(ergas)


def compute_mse_per_band(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = _squeeze_hw_c(gt)
    pred = _squeeze_hw_c(pred)
    diff2 = (gt - pred) ** 2
    return np.mean(diff2, axis=(0, 1))


@dataclass
class ImageMetrics:
    name: str
    ssim: float
    sam: float
    ergas: float


def iter_test_names(cmhf_root: str) -> list[str]:
    list_path = os.path.join(cmhf_root, "CAVEdata", "List")
    d = sio.loadmat(list_path)
    ind = d["Ind"].astype(int).ravel().tolist()  # 1-based indices, length 32

    # Use the last 12 as test indices (matches CMHF-net's original split logic)
    test_ind = [i - 1 for i in ind[20:]]

    x_dir = os.path.join(cmhf_root, "CAVEdata", "X")
    names = sorted(os.listdir(x_dir))
    # Map indices into sorted listdir ordering (same as original code behavior)
    return [names[i] for i in test_ind]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmhf_root", required=True, help="Path to methods/_MHFnet/CMHF-net")
    ap.add_argument("--pred_dir", required=True, help="Folder containing per-image .mat outputs with key 'outX'")
    ap.add_argument("--sf", type=int, default=32, help="Scale factor used by ERGAS (CMHF uses 32)")
    args = ap.parse_args()

    cmhf_root = args.cmhf_root
    pred_dir = args.pred_dir
    sf = int(args.sf)

    test_names = iter_test_names(cmhf_root)
    if len(test_names) != 12:
        raise RuntimeError(f"Expected 12 test images from split, got {len(test_names)}")

    all_metrics: list[ImageMetrics] = []
    for name in test_names:
        gt_path = os.path.join(cmhf_root, "CAVEdata", "X", name)
        pred_path = os.path.join(pred_dir, name)

        gt = sio.loadmat(gt_path)["msi"]
        pred = sio.loadmat(pred_path)["outX"]
        gt = _squeeze_hw_c(gt)
        pred = _squeeze_hw_c(pred)

        mse_per_band = compute_mse_per_band(gt, pred)
        m = ImageMetrics(
            name=name,
            ssim=compute_ms_ssim(pred, gt),
            sam=compute_sam(pred, gt),
            ergas=compute_ergas_from_mse(mse_per_band, pred, sf=sf),
        )
        all_metrics.append(m)
        print(f"{name}: SSIM={m.ssim:.6f} SAM={m.sam:.6f} ERGAS={m.ergas:.6f}")

    ssim_avg = float(np.mean([m.ssim for m in all_metrics]))
    sam_avg = float(np.mean([m.sam for m in all_metrics]))
    ergas_avg = float(np.mean([m.ergas for m in all_metrics]))
    print("==== Averages over 12 CAVE test images ====")
    print(f"SSIM={ssim_avg:.6f} SAM={sam_avg:.6f} ERGAS={ergas_avg:.6f}")


if __name__ == "__main__":
    main()
