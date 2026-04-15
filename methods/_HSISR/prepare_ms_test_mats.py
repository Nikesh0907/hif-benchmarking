#!/usr/bin/env python3
"""Prepare HSISR test .mat files (gt/ms/ms_bicubic) in pure Python.

HSISR's original code (methods/_HSISR/data/load_test_data.py) expects each test
sample as a .mat file containing:
  - gt: HR HSI cube (H,W,31)
  - ms: LR HSI cube (H/sf,W/sf,31)
  - ms_bicubic: bicubic-upsampled ms back to (H,W,31)

The authors provide MATLAB scripts under methods/_HSISR/matlab_code that do this
with imresize (bicubic). This script reproduces that pipeline so you can run in
Kaggle without MATLAB.

Typical usage (CAVE-like, gt stored under key 'hsi'):
  python methods/_HSISR/prepare_ms_test_mats.py \
    --hsi_dir /kaggle/input/yourdata/Test/HSI \
    --out_dir /kaggle/working/hsisr_test_x4 \
    --gt_key hsi --sf 4

Then run:
  python methods/_HSISR/test_ms_mats.py --mat_dir /kaggle/working/hsisr_test_x4 \
    --weights methods/_HSISR/models/Cave_DeepShare_...pth --sf 4
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import scipy.io

from tools.hif_metrics import normalize01


def _ensure_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {x.shape}")
    # If it's CHW, convert.
    if x.shape[0] in (31, 34) and x.shape[2] not in (31, 34):
        return np.transpose(x, (1, 2, 0))
    return x


def _resize_hsi_bicubic(hsi_hwc: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    h, w, c = hsi_hwc.shape
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    for b in range(c):
        out[:, :, b] = cv2.resize(hsi_hwc[:, :, b], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare HSISR gt/ms/ms_bicubic test mats")
    ap.add_argument("--hsi_dir", required=True, help="Directory containing GT HSI mats")
    ap.add_argument("--out_dir", required=True, help="Output directory for HSISR-style mats")
    ap.add_argument("--gt_key", default="hsi", help="Key for GT cube in the input mats (e.g. hsi or ref)")
    ap.add_argument("--sf", type=int, default=4, help="Scale factor (paper commonly reports x4)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mats = sorted(glob.glob(os.path.join(args.hsi_dir, "*.mat")))
    if not mats:
        raise SystemExit(f"No .mat files found in {args.hsi_dir}")
    if args.limit and args.limit > 0:
        mats = mats[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    for idx, p in enumerate(mats, start=1):
        name = Path(p).stem
        m = scipy.io.loadmat(p)
        if args.gt_key not in m:
            raise SystemExit(f"Missing gt_key '{args.gt_key}' in {p}. Keys={list(m.keys())}")

        gt = _ensure_hwc(m[args.gt_key]).astype(np.float32)
        gt = normalize01(gt)

        h, w, c = gt.shape
        if c != 31:
            raise SystemExit(f"Expected 31 bands, got {c} in {p}")
        if h % args.sf != 0 or w % args.sf != 0:
            raise SystemExit(f"Image {name} shape {h}x{w} not divisible by sf={args.sf}")

        ms = _resize_hsi_bicubic(gt, (h // args.sf, w // args.sf))
        ms_bicubic = _resize_hsi_bicubic(ms, (h, w))

        out_path = os.path.join(args.out_dir, f"{name}.mat")
        scipy.io.savemat(out_path, {"gt": gt.astype(np.float32), "ms": ms.astype(np.float32), "ms_bicubic": ms_bicubic.astype(np.float32)})
        if (idx % 25) == 0 or idx == len(mats):
            print(f"[{idx}/{len(mats)}] wrote {out_path}")

    print(f"Done. Wrote {len(mats)} mats to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
