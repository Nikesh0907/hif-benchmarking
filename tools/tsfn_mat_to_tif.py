#!/usr/bin/env python3
"""Convert a directory of .mat files into TSFN-compatible .tif files.

TSFN expects test files as TIFF where:
  - shape is (34, H, W)
  - bands 0..30 are HSI (31 bands)
  - bands 31..33 are RGB (3 bands)

This script tries to read common keys used in this repo:
  - HSI: 'hsi' (CAVE), or 'ref' (Harvard)
  - RGB/MSI: 'msi' (CAVE/Harvard simulated Nikon D700)

If no RGB/MSI is present, you can synthesize RGB from HSI using --rgb_from_hsi.

Example (CAVE mats containing hsi+msi):
  python tools/tsfn_mat_to_tif.py \
    --mat_dir /kaggle/input/cave-dataset-2/Data/Test \
    --out_dir methods/_TSFN/data/test \
    --hsi_key hsi \
    --rgb_key msi

Then run TSFN:
  cd methods/_TSFN
  python test.py --model_path ./models/ --results_path ./results/
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io
import tifffile
import cv2


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / np.iinfo(arr.dtype).max).clip(0.0, 1.0)

    arr = arr.astype(np.float32)
    maxv = float(np.nanmax(arr)) if arr.size else 1.0
    minv = float(np.nanmin(arr)) if arr.size else 0.0

    if maxv <= 1.0 and minv >= 0.0:
        return arr
    if maxv > 1.0 and minv >= 0.0:
        return (arr / maxv).clip(0.0, 1.0)

    denom = (maxv - minv) if (maxv - minv) != 0 else 1.0
    return ((arr - minv) / denom).clip(0.0, 1.0)


def _ensure_hwc(x: np.ndarray) -> np.ndarray:
    """Ensure array is HxWxC."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={x.shape}")

    # Common cases:
    # - HxWxC (repo)
    # - CxHxW (some code)
    h, w, c = x.shape
    if c in (3, 31, 34):
        return x

    c2, h2, w2 = x.shape
    if c2 in (3, 31, 34) and h2 > 1 and w2 > 1:
        return np.transpose(x, (1, 2, 0))

    # Fallback: assume already HWC
    return x


def _synth_rgb_from_hsi(hsi_hwc: np.ndarray, rgb_bands: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = rgb_bands
    if max(rgb_bands) >= hsi_hwc.shape[2]:
        raise ValueError(f"rgb_bands {rgb_bands} out of range for HSI with C={hsi_hwc.shape[2]}")
    rgb = np.stack([hsi_hwc[:, :, r], hsi_hwc[:, :, g], hsi_hwc[:, :, b]], axis=2)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert .mat cubes to TSFN-compatible .tif")
    ap.add_argument("--mat_dir", required=True, help="Directory with .mat files")
    ap.add_argument(
        "--rgb_mat_dir",
        default="",
        help=(
            "Optional directory containing matching per-scene .mat files with RGB/MSI (same basename). "
            "Useful for Harvard where GT mats contain only 'ref' and MSI is stored separately under data/MS/Harvard."
        ),
    )
    ap.add_argument("--out_dir", required=True, help="Output directory for .tif files")
    ap.add_argument("--hsi_key", default="hsi", help="Key for HSI cube (e.g. hsi or ref)")
    ap.add_argument("--rgb_key", default="msi", help="Key for RGB/MSI cube (e.g. msi or rgb)")
    ap.add_argument("--rgb_from_hsi", action="store_true", help="If RGB key missing, synthesize RGB from HSI")
    ap.add_argument(
        "--rgb_bands",
        default="20,10,5",
        help="Bands to use when synthesizing RGB from HSI (comma-separated indices)",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mats = sorted(glob.glob(os.path.join(args.mat_dir, "*.mat")))
    if not mats:
        raise SystemExit(f"No .mat files found in: {args.mat_dir}")

    if args.limit and args.limit > 0:
        mats = mats[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    rgb_bands = tuple(int(x.strip()) for x in args.rgb_bands.split(","))
    if len(rgb_bands) != 3:
        raise SystemExit("--rgb_bands must have 3 comma-separated ints, e.g. 20,10,5")

    converted = 0
    for i, mat_path in enumerate(mats, start=1):
        name = Path(mat_path).stem
        m = scipy.io.loadmat(mat_path)

        if args.hsi_key not in m:
            raise SystemExit(f"Missing HSI key '{args.hsi_key}' in {mat_path}. Keys={list(m.keys())}")
        hsi = _ensure_hwc(np.asarray(m[args.hsi_key]))

        rgb: Optional[np.ndarray] = None
        if args.rgb_key in m:
            rgb = _ensure_hwc(np.asarray(m[args.rgb_key]))
        elif args.rgb_mat_dir:
            rgb_mat_path = os.path.join(args.rgb_mat_dir, name + ".mat")
            if os.path.isfile(rgb_mat_path):
                m_rgb = scipy.io.loadmat(rgb_mat_path)
                if args.rgb_key in m_rgb:
                    rgb = _ensure_hwc(np.asarray(m_rgb[args.rgb_key]))
                else:
                    # Common alternative key
                    if "rgb" in m_rgb and args.rgb_key != "rgb":
                        rgb = _ensure_hwc(np.asarray(m_rgb["rgb"]))
            if rgb is None:
                # Try image files (common Kaggle layout)
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    img_path = os.path.join(args.rgb_mat_dir, name + ext)
                    if not os.path.isfile(img_path):
                        continue
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 2:
                        img = np.repeat(img[:, :, None], 3, axis=2)
                    # OpenCV loads BGR -> RGB
                    if img.shape[2] >= 3:
                        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
                    rgb = _ensure_hwc(img.astype(np.float32))
                    break
        elif args.rgb_from_hsi:
            rgb = _synth_rgb_from_hsi(hsi, rgb_bands)

        if rgb is None:
            raise SystemExit(
                f"Missing RGB key '{args.rgb_key}' in {mat_path} and --rgb_from_hsi not set. "
                f"Keys={list(m.keys())}"
            )

        # Normalize to [0,1]
        hsi01 = _normalize01(hsi)
        rgb01 = _normalize01(rgb)

        # Ensure shapes match spatially
        if hsi01.shape[0] != rgb01.shape[0] or hsi01.shape[1] != rgb01.shape[1]:
            raise SystemExit(
                f"Spatial mismatch in {name}: HSI={hsi01.shape} RGB={rgb01.shape}. "
                f"Provide matching msi/rgb, or pre-resize before conversion."
            )

        if hsi01.shape[2] != 31:
            raise SystemExit(f"HSI must have 31 bands for TSFN. {name} has C={hsi01.shape[2]}")
        if rgb01.shape[2] != 3:
            raise SystemExit(f"RGB/MSI must have 3 channels for TSFN. {name} has C={rgb01.shape[2]}")

        # Stack to (34,H,W) as TSFN expects
        stacked_hwc = np.concatenate([hsi01, rgb01], axis=2)  # HxWx34
        stacked_chw = np.transpose(stacked_hwc, (2, 0, 1))  # 34xHxW

        # Save as uint16 to be friendly with torchvision ToTensor scaling
        stacked_u16 = (stacked_chw * 65535.0).round().clip(0, 65535).astype(np.uint16)

        out_path = os.path.join(args.out_dir, name + ".tif")
        tifffile.imwrite(out_path, stacked_u16)
        converted += 1

        if (i % 25) == 0 or i == len(mats):
            print(f"[{i}/{len(mats)}] converted {converted} -> {out_path}")

    print(f"Done. Wrote {converted} tif files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
