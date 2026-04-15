#!/usr/bin/env python3
"""Test HSISR (DeepShare) on HSISR-style .mat files (gt/ms/ms_bicubic).

This matches the original HSISR dataloader format:
  - gt: HR HSI (H,W,31)
  - ms: LR HSI (H/sf,W/sf,31)
  - ms_bicubic: bicubic-upsampled ms to (H,W,31)

Use prepare_ms_test_mats.py to generate these mats from your GT HSI.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io
import torch

from methods._HSISR.BlockModule import DeepShare
from methods._HSISR.basicModule import default_conv
from tools.hif_metrics import compute_metrics, normalize01


def _load_sample(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = scipy.io.loadmat(path)
    for k in ("gt", "ms", "ms_bicubic"):
        if k not in m:
            raise KeyError(f"Missing key '{k}' in {path}. Keys={list(m.keys())}")
    gt = np.asarray(m["gt"], dtype=np.float32)
    ms = np.asarray(m["ms"], dtype=np.float32)
    lms = np.asarray(m["ms_bicubic"], dtype=np.float32)
    return gt, ms, lms


def _to_torch_chw(arr_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    arr01 = normalize01(arr_hwc).astype(np.float32)
    return torch.from_numpy(arr01).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description="HSISR test runner for gt/ms/ms_bicubic mats")
    ap.add_argument("--mat_dir", required=True, help="Directory with .mat files (keys gt, ms, ms_bicubic)")
    ap.add_argument("--weights", required=True, help="Path to HSISR .pth weights")
    ap.add_argument("--sf", type=int, default=4, help="Scale factor used to generate ms")
    ap.add_argument("--cuda", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)

    # Must match pretrained weights (these defaults match bundled CAVE/Harvard models).
    ap.add_argument("--n_feats", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--n_subs", type=int, default=8)
    ap.add_argument("--n_ovls", type=int, default=2)
    ap.add_argument("--n_colors", type=int, default=31)

    args = ap.parse_args()

    device = torch.device("cuda" if (args.cuda == 1 and torch.cuda.is_available()) else "cpu")

    mats = sorted(glob.glob(os.path.join(args.mat_dir, "*.mat")))
    if not mats:
        raise SystemExit(f"No .mat files found in {args.mat_dir}")
    if args.limit and args.limit > 0:
        mats = mats[: args.limit]

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

    sums: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "sam": 0.0, "ergas": 0.0}

    with torch.no_grad():
        for idx, p in enumerate(mats, start=1):
            name = Path(p).stem
            gt, ms, lms = _load_sample(p)

            if gt.ndim != 3 or ms.ndim != 3 or lms.ndim != 3:
                raise SystemExit(f"Bad shapes in {p}: gt={gt.shape} ms={ms.shape} lms={lms.shape}")

            x = _to_torch_chw(ms, device)
            lms_t = _to_torch_chw(lms, device)

            pred = model(x, lms_t, modality="spectral")
            pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

            m = compute_metrics(gt, pred, ratio=args.sf)
            for k in sums:
                sums[k] += m[k]

            print(
                f"[{idx}/{len(mats)}] {name} PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} SAM={m['sam']:.4f} ERGAS={m['ergas']:.3f}"
            )

    n = float(len(mats))
    avg = {k: v / n for k, v in sums.items()}
    print(
        f"\nAVERAGE (sf={args.sf}, n={len(mats)}): PSNR={avg['psnr']:.3f} SSIM={avg['ssim']:.4f} SAM={avg['sam']:.4f} ERGAS={avg['ergas']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
