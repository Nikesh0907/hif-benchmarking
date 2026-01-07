#!/usr/bin/env python3
"""Evaluate CMHF-net (MHF-net) outputs on the CAVE test split.

This script computes metrics using the same formula style used by DBIN
(see methods/_DBIN/dbintest.py):
- SSIM: average SSIM across bands (data_range=1.0)
- SAM: mean spectral angle (degrees) across pixels
- ERGAS: 100/sf * sqrt(mean(mse_band / (mean(out_band)^2 + eps)))

It assumes you already ran CMHF-net inference to produce per-image .mat files
containing key `outX` under `pred_dir`.

Typical workflow:
1) Prepare CMHF-net CAVEdata/*
2) Run CMHF-net `CAVEmain.py --mode=testAll`
3) Run this script

Example:
  python methods/_MHFnet/eval_mhfnet_cave.py \
    --pred_dir methods/_MHFnet/CMHF-net/TestResult/Result \
    --gt_dir methods/_MHFnet/CMHF-net/CAVEdata/X \
    --split_list methods/_MHFnet/CMHF-net/CAVEdata/List
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio


def _ssim2d(a: np.ndarray, b: np.ndarray) -> float:
    """Single-band SSIM with data_range=1.0 (float images)."""
    try:
        from skimage.metrics import structural_similarity as ssim

        return float(ssim(a, b, data_range=1.0))
    except Exception:
        # Older skimage
        from skimage.measure import compare_ssim as ssim  # type: ignore

        return float(ssim(a, b, data_range=1.0))


def compute_ms_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]

    if image1.shape != image2.shape:
        raise ValueError(f"SSIM shape mismatch: {image1.shape} vs {image2.shape}")

    n_bands = image1.shape[2]
    ms_ssim = 0.0
    for i in range(n_bands):
        ms_ssim += _ssim2d(image1[:, :, i], image2[:, :, i])
    return ms_ssim / float(n_bands)


def compute_sam(image1: np.ndarray, image2: np.ndarray) -> float:
    """DBIN-style SAM: mean arccos over pixels, in degrees."""
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]

    if image1.shape != image2.shape:
        raise ValueError(f"SAM shape mismatch: {image1.shape} vs {image2.shape}")

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
    """DBIN-style ERGAS taking per-band MSE and mean(out) per band."""
    out = np.asarray(out)
    if out.ndim == 4:
        out = out[0]

    h, w, c = out.shape
    out2 = np.reshape(out, (h * w, c))
    out_mean = np.mean(out2, axis=0).reshape((c, 1))

    mse_per_band = np.asarray(mse_per_band).reshape((c, 1))
    ergas = 100.0 / float(sf) * np.sqrt(np.mean(mse_per_band / (out_mean**2 + 1e-12)))
    return float(ergas)


def mse_per_band(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 4:
        pred = pred[0]

    if gt.shape != pred.shape:
        raise ValueError(f"MSE shape mismatch: {gt.shape} vs {pred.shape}")

    return np.mean((gt - pred) ** 2, axis=(0, 1))


@dataclass(frozen=True)
class Item:
    name: str
    gt_path: str
    pred_path: str


def load_test_names_from_list(split_list_path: str, all_names_sorted: list[str]) -> list[str]:
    """Use CMHF-net's CAVEdata/List Ind to pick test scene names.

    For 32 scenes: uses the last 12 (indices 20:32, standard CAVE split).
    For 20 scenes: uses the last 12 (indices 8:20).
    For N scenes: uses the last 12 (or fewer if N < 12).

    NOTE: Ind indexes depend on file ordering in the original MATLAB/Python code.
    We use sorted filenames for determinism.
    """
    mat = sio.loadmat(split_list_path)
    if "Ind" not in mat:
        raise KeyError(f"Missing 'Ind' in {split_list_path}")
    ind = mat["Ind"].reshape(-1).astype(int)
    n_total = ind.size

    # Use the last 12 indices (or fewer if less than 12 total).
    n_test = min(12, n_total)
    test_idx = ind[n_total - n_test:n_total] - 1  # 1-based -> 0-based
    out: list[str] = []
    for i in test_idx:
        if i < 0 or i >= len(all_names_sorted):
            raise IndexError(
                f"Index {i} out of range for {len(all_names_sorted)} files. "
                f"Your GT folder and List file may not match."
            )
        out.append(all_names_sorted[i])
    print(f"[eval] Using {len(out)} test scenes (last {len(out)} of {n_total} total)")
    return out


def build_items(gt_dir: str, pred_dir: str, split_list: str | None) -> list[Item]:
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.mat")))
    if not gt_paths:
        raise FileNotFoundError(f"No .mat files found in gt_dir={gt_dir}")

    all_names_sorted = [Path(p).name for p in gt_paths]

    if split_list:
        test_names = load_test_names_from_list(split_list, all_names_sorted)
    else:
        # Fallback: use last 12 alphabetically.
        test_names = all_names_sorted[-12:]

    items: list[Item] = []
    for name in test_names:
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)
        items.append(Item(name=name, gt_path=gt_path, pred_path=pred_path))
    return items


def _load_gt_hsi(path: str) -> np.ndarray:
    d = sio.loadmat(path)
    if "msi" in d:
        arr = np.asarray(d["msi"], dtype=np.float64)
    elif "hsi" in d:
        arr = np.asarray(d["hsi"], dtype=np.float64)
    else:
        raise KeyError(f"Expected key 'msi' or 'hsi' in GT mat {path}")
    return np.clip(arr, 0.0, 1.0)


def _load_pred_hsi(path: str) -> np.ndarray:
    d = sio.loadmat(path)
    # CMHF-net testAll saves outX.
    if "outX" in d:
        arr = np.asarray(d["outX"], dtype=np.float64)
        if arr.ndim == 4:
            arr = arr[0]
        return np.clip(arr, 0.0, 1.0)

    # Some repos save as 'sri' or 'res'
    for k in ("sri", "res", "RE"):
        if k in d:
            arr = np.asarray(d[k], dtype=np.float64)
            if arr.ndim == 4:
                arr = arr[0]
            return np.clip(arr, 0.0, 1.0)

    raise KeyError(f"No supported prediction key found in {path} (expected outX/sri/res/RE)")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_dir",
        default="methods/_MHFnet/CMHF-net/TestResult/Result",
        help="Folder with predicted .mat outputs (expects key outX)",
    )
    ap.add_argument(
        "--gt_dir",
        default="methods/_MHFnet/CMHF-net/CAVEdata/X",
        help="Folder with GT .mat files (expects key msi)",
    )
    ap.add_argument(
        "--split_list",
        default="methods/_MHFnet/CMHF-net/CAVEdata/List",
        help="Optional CMHF-net split list mat (key Ind). Set to empty to disable.",
    )
    ap.add_argument("--sf", type=int, default=32, help="Scale factor for ERGAS (CMHF-net uses 32 for Z generation)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    split_list = args.split_list.strip() or None

    items = build_items(args.gt_dir, args.pred_dir, split_list)

    ssim_list: list[float] = []
    sam_list: list[float] = []
    ergas_list: list[float] = []

    print("name,ssim,sam,ergas")
    for it in items:
        if not os.path.exists(it.pred_path):
            raise FileNotFoundError(
                f"Missing prediction for {it.name}: expected {it.pred_path}. "
                f"Run CMHF-net testAll first or point --pred_dir correctly."
            )

        gt = _load_gt_hsi(it.gt_path)
        pred = _load_pred_hsi(it.pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {it.name}: gt={gt.shape}, pred={pred.shape}")

        ssim_v = compute_ms_ssim(pred, gt)
        sam_v = compute_sam(pred, gt)
        mse_v = mse_per_band(gt, pred)
        ergas_v = compute_ergas_from_mse(mse_v, pred, sf=args.sf)

        ssim_list.append(ssim_v)
        sam_list.append(sam_v)
        ergas_list.append(ergas_v)

        print(f"{it.name},{ssim_v:.6f},{sam_v:.6f},{ergas_v:.6f}")

    print("avg,{:.6f},{:.6f},{:.6f}".format(np.mean(ssim_list), np.mean(sam_list), np.mean(ergas_list)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
