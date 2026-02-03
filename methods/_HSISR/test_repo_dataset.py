#!/usr/bin/env python3
"""Test HSISR (DeepShare) on this repo's processed datasets.

Reads:
  - data/GT/<dataset>/<name>.mat: GT HSI under key 'hsi' (CAVE) or 'ref' (Harvard)
  - data/HS/<dataset>/<sf>/<name>.mat: LR HSI under key 'hsi'

Runs the model and prints average PSNR/SSIM/SAM/ERGAS.

Example:
  python methods/_HSISR/test_repo_dataset.py --dataset CAVE --sf 4 \
    --weights methods/_HSISR/models/Cave_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Wed_Mar_31_03:00:46_2021.pth

Kaggle tip:
  pip install -r auxiliary/requirements.txt
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or cpu wheel
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import scipy.io
import torch

from methods._HSISR.BlockModule import DeepShare
from methods._HSISR.basicModule import default_conv
from tools.hif_metrics import compute_metrics, normalize01


def _load_gt_hsi(gt_mat_path: str) -> np.ndarray:
    mat = scipy.io.loadmat(gt_mat_path)
    if "hsi" in mat:
        return np.asarray(mat["hsi"])
    if "ref" in mat:
        return np.asarray(mat["ref"])
    raise KeyError(f"No GT HSI key found in {gt_mat_path}. Keys={list(mat.keys())}")


def _load_lr_hsi(lr_mat_path: str) -> np.ndarray:
    mat = scipy.io.loadmat(lr_mat_path)
    if "hsi" not in mat:
        raise KeyError(f"No LR HSI key 'hsi' in {lr_mat_path}. Keys={list(mat.keys())}")
    return np.asarray(mat["hsi"])


def _bicubic_upsample_hsi(lr_hsi: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    bands = lr_hsi.shape[2]
    out = []
    for b in range(bands):
        out.append(cv2.resize(lr_hsi[:, :, b], (w, h), interpolation=cv2.INTER_CUBIC))
    return np.stack(out, axis=2)


def _to_torch_chw(arr_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    arr01 = normalize01(arr_hwc).astype(np.float32)
    return torch.from_numpy(arr01).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description="HSISR test runner for repo datasets")
    ap.add_argument("--dataset", required=True, choices=["CAVE", "Harvard"], help="Dataset name")
    ap.add_argument("--sf", type=int, default=4, help="Downsample scale factor (4/8/16)")
    ap.add_argument("--data_root", type=str, default="data", help="Repo data root")
    ap.add_argument("--weights", required=True, type=str, help="Path to .pth weights")
    ap.add_argument("--cuda", type=int, default=1, help="1 to use CUDA if available")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0=all)")
    ap.add_argument("--out_dir", type=str, default="", help="Optional directory to save outputs as .mat (key 'sri')")

    # Model hyperparams must match pretrained weights naming; defaults match bundled weights.
    ap.add_argument("--n_feats", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--n_subs", type=int, default=8)
    ap.add_argument("--n_ovls", type=int, default=2)
    ap.add_argument("--n_colors", type=int, default=31)

    args = ap.parse_args()

    device = torch.device("cuda" if (args.cuda == 1 and torch.cuda.is_available()) else "cpu")

    gt_dir = Path(args.data_root) / "GT" / args.dataset
    hs_dir = Path(args.data_root) / "HS" / args.dataset / str(args.sf)

    gt_mats = sorted(glob.glob(str(gt_dir / "*.mat")))
    if not gt_mats:
        raise SystemExit(f"No GT mats found in {gt_dir}")

    if args.limit and args.limit > 0:
        gt_mats = gt_mats[: args.limit]

    # Build model
    model = DeepShare(
        n_subs=args.n_subs,
        n_ovls=args.n_ovls,
        n_colors=args.n_colors,
        n_blocks=args.n_blocks,
        n_feats=args.n_feats,
        n_scale=args.sf,
        res_scale=0.1,
        use_share=True,
        conv=default_conv,
    )
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    sums: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "sam": 0.0, "ergas": 0.0}

    with torch.no_grad():
        for idx, gt_path in enumerate(gt_mats, start=1):
            name = Path(gt_path).stem
            lr_path = hs_dir / f"{name}.mat"
            if not lr_path.exists():
                raise SystemExit(f"Missing LR HSI mat: {lr_path} (run dataset script first)")

            gt_hsi = _load_gt_hsi(gt_path)
            lr_hsi = _load_lr_hsi(str(lr_path))

            # Ensure [H,W,C]
            if gt_hsi.ndim != 3:
                raise SystemExit(f"Unexpected GT shape for {name}: {gt_hsi.shape}")
            if lr_hsi.ndim != 3:
                raise SystemExit(f"Unexpected LR shape for {name}: {lr_hsi.shape}")

            h, w, _ = gt_hsi.shape
            lms = _bicubic_upsample_hsi(lr_hsi, (h, w))

            x = _to_torch_chw(lr_hsi, device)
            lms_t = _to_torch_chw(lms, device)

            pred = model(x, lms_t, modality="spectral")
            pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

            m = compute_metrics(gt_hsi, pred, ratio=args.sf)
            for k in sums:
                sums[k] += m[k]

            if args.out_dir:
                scipy.io.savemat(str(Path(args.out_dir) / f"{name}.mat"), {"sri": normalize01(pred)})

            print(
                f"[{idx}/{len(gt_mats)}] {args.dataset}/{name} "
                f"PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} SAM={m['sam']:.4f} ERGAS={m['ergas']:.3f}"
            )

    n = float(len(gt_mats))
    avg = {k: v / n for k, v in sums.items()}
    print(
        f"\nAVERAGE ({args.dataset}, sf={args.sf}, n={len(gt_mats)}): "
        f"PSNR={avg['psnr']:.3f} SSIM={avg['ssim']:.4f} SAM={avg['sam']:.4f} ERGAS={avg['ergas']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
