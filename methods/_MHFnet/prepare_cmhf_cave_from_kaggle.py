#!/usr/bin/env python3
"""Prepare CMHF-net 'CAVEdata' inputs from Kaggle CAVE .mat files.

CMHF-net expects the following folders (relative to CMHF-net root):
- CAVEdata/X/<name>.mat with key 'msi' (H,W,31) in [0,1]
- CAVEdata/Y/<name>.mat with key 'RGB' (H,W,3) in [0,1]
- CAVEdata/Z/<name>.mat with key 'Zmsi' (H/32,W/32,31) in [0,1]
- CAVEdata/List with key 'Ind' (1-based indices; first 20 train, last 12 test)
- CAVEdata/iniA (key 'iniA') and CAVEdata/iniUp (key 'iniUp1')

This script builds those from:
- HSI mats (key 'hsi' or 'msi')
- optional RGB mats (key 'rgb' or 'msi' or 'RGB')

If RGB mats are not provided, RGB is synthesized via spectral response matrix R
loaded from --response_mat (expects key 'R').

Example (Kaggle):
  python methods/_MHFnet/prepare_cmhf_cave_from_kaggle.py \
    --hsi_dir /kaggle/input/cave-dataset-2/Data/Train/HSI \
    --rgb_dir /kaggle/input/cave-dataset-2/Data/Train/RGB \
    --cmhf_root methods/_MHFnet/CMHF-net \
    --response_mat methods/_MHFnet/CMHF-net/rowData/CAVEdata/response\ coefficient.mat
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
import scipy.io as sio


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        return (x.astype(np.float32) / np.iinfo(x.dtype).max).clip(0.0, 1.0)
    x = x.astype(np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 1.0:
        return np.clip(x, 0.0, 1.0)
    if mx <= 255.0:
        denom = 255.0
    elif mx <= 4095.0:
        denom = 4095.0
    elif mx <= 65535.0:
        denom = 65535.0
    else:
        denom = mx
    return np.clip(x / denom, 0.0, 1.0)


def _load_hsi(path: str) -> np.ndarray:
    d = sio.loadmat(path)
    if "hsi" in d:
        arr = d["hsi"]
    elif "msi" in d:
        arr = d["msi"]
    else:
        raise KeyError(f"Expected 'hsi' or 'msi' in {path}")
    arr = _normalize01(np.asarray(arr))
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"HSI must be HxWxC; got {arr.shape} in {path}")
    if arr.shape[2] < 31:
        raise ValueError(f"HSI must have >=31 bands; got {arr.shape} in {path}")
    return arr[:, :, :31]


def _load_rgb(path: str) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    d = sio.loadmat(path)
    for k in ("rgb", "RGB", "msi"):
        if k in d:
            arr = _normalize01(np.asarray(d[k]))
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            return arr
    return None


def _compute_rgb_from_R(hsi: np.ndarray, R: np.ndarray) -> np.ndarray:
    # CMHF reader uses: Y = tensordot(X, R, (2,0))
    R = np.asarray(R)
    if R.shape == (31, 3):
        pass
    elif R.shape == (3, 31):
        R = R.T
    else:
        raise ValueError(f"Unexpected R shape {R.shape}; expected 31x3 (or 3x31)")
    rgb = np.tensordot(hsi, R, axes=(2, 0))
    return np.clip(rgb.astype(np.float32), 0.0, 1.0)


def _compute_Z_from_C(hsi: np.ndarray, C: np.ndarray, sf: int = 32) -> np.ndarray:
    # Matches CMHF-net PrepareDataAndiniValue loop.
    H, W, B = hsi.shape
    if H % sf != 0 or W % sf != 0:
        raise ValueError(f"HSI size must be divisible by {sf}; got {(H, W)}")
    C = np.asarray(C, dtype=np.float32)
    if C.shape != (sf, sf):
        raise ValueError(f"Expected C to have shape {(sf, sf)}, got {C.shape}")
    z = np.zeros((H // sf, W // sf, B), dtype=np.float32)
    for j in range(sf):
        for k in range(sf):
            z += hsi[j:H:sf, k:W:sf, :] * C[k, j]
    return np.clip(z, 0.0, 1.0)


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsi_dir", required=True, help="Folder with CAVE HSI mats")
    ap.add_argument("--rgb_dir", default="", help="Optional folder with RGB mats (same basename).")
    ap.add_argument("--cmhf_root", required=True, help="Path to methods/_MHFnet/CMHF-net")
    ap.add_argument(
        "--response_mat",
        required=True,
        help="Mat file containing keys R and C (e.g. CMHF-net/rowData/CAVEdata/response coefficient.mat)",
    )
    ap.add_argument("--limit", type=int, default=32, help="Number of scenes to use (default 32)")
    ap.add_argument("--paper_split", action="store_true", help="Use CMHF paper split Ind list (only if your file ordering matches).")
    args = ap.parse_args()

    hsi_paths = sorted(glob(os.path.join(args.hsi_dir, "*.mat")))
    if not hsi_paths:
        raise SystemExit(f"No .mat files found in {args.hsi_dir}")

    limit = min(int(args.limit), len(hsi_paths))
    hsi_paths = hsi_paths[:limit]

    resp = sio.loadmat(args.response_mat)
    if "R" not in resp or "C" not in resp:
        raise SystemExit(f"response_mat must contain 'R' and 'C' keys: {args.response_mat}")
    R = resp["R"]
    C = resp["C"]

    cave_root = os.path.join(args.cmhf_root, "CAVEdata")
    x_dir = os.path.join(cave_root, "X")
    y_dir = os.path.join(cave_root, "Y")
    z_dir = os.path.join(cave_root, "Z")
    _mkdir(x_dir)
    _mkdir(y_dir)
    _mkdir(z_dir)

    # Build Ind list
    if args.paper_split:
        ind = np.array(
            [2, 31, 25, 6, 27, 15, 19, 14, 12, 28, 26, 29, 8, 13, 22, 7, 24, 30, 10, 23, 18, 17, 21, 3, 9, 4, 20, 5, 16, 32, 11, 1],
            dtype=np.int32,
        )
        if limit != 32:
            raise SystemExit("--paper_split requires --limit 32 and exactly 32 scenes")
    else:
        ind = np.arange(1, limit + 1, dtype=np.int32)

    # Prepare iniA accumulators over first 20 train scenes (or fewer if limit<20)
    n_train = min(20, limit)
    yty = np.zeros((3, 3), dtype=np.float64)
    ytx = np.zeros((3, 31), dtype=np.float64)

    for idx, p in enumerate(hsi_paths, start=1):
        base = os.path.splitext(os.path.basename(p))[0]
        hsi = _load_hsi(p)

        rgb = None
        rgb_dir = args.rgb_dir.strip()
        if rgb_dir:
            rgb = _load_rgb(os.path.join(rgb_dir, base + ".mat"))
        if rgb is None:
            rgb = _compute_rgb_from_R(hsi, R)

        z = _compute_Z_from_C(hsi, C, sf=32)

        sio.savemat(os.path.join(x_dir, base + ".mat"), {"msi": hsi})
        sio.savemat(os.path.join(y_dir, base + ".mat"), {"RGB": rgb})
        sio.savemat(os.path.join(z_dir, base + ".mat"), {"Zmsi": z})

        # Incremental iniA stats (train split uses first 20 indices)
        if idx <= n_train:
            Y = rgb.reshape((-1, 3)).astype(np.float64)
            X = hsi.reshape((-1, 31)).astype(np.float64)
            yty += Y.T @ Y
            ytx += Y.T @ X

        print(f"[{idx}/{limit}] wrote {base}.mat")

    iniA = np.linalg.pinv(yty) @ ytx
    sio.savemat(os.path.join(cave_root, "iniA"), {"iniA": iniA})

    iniUp1 = np.tile(np.eye(31, dtype=np.float32), (3, 3, 1, 1))
    sio.savemat(os.path.join(cave_root, "iniUp"), {"iniUp1": iniUp1})

    sio.savemat(os.path.join(cave_root, "List"), {"Ind": ind.reshape((1, -1))})

    print("Done. Created CMHF-net CAVEdata/{X,Y,Z,List,iniA,iniUp}.")


if __name__ == "__main__":
    main()
