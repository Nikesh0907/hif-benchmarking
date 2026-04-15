#!/usr/bin/env python3
"""Test HSISR on Kaggle-format dataset (separate HSI and RGB directories).

Usage:
    python methods/_HSISR/test_kaggle_data.py \
        /kaggle/input/cave-dataset-2/Data/Test/HSI \
        --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
        --sf 4 \
        --weights methods/_HSISR/models/Cave_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Wed_Mar_31_03:00:46_2021.pth
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch

# Add repo root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from methods._HSISR.BlockModule import DeepShare
from tools.hif_metrics import compute_metrics, normalize01


def load_mat(path, key=None):
    """Load .mat file, auto-detect key if not specified."""
    mat = scipy.io.loadmat(path)
    if key:
        return np.asarray(mat[key], dtype=np.float32)
    # Try common keys
    for k in ['hsi', 'gt', 'ref', 'cube', 'X']:
        if k in mat:
            return np.asarray(mat[k], dtype=np.float32)
    # Fallback: largest 3D array
    for k in mat.keys():
        if not k.startswith('__'):
            arr = np.asarray(mat[k])
            if arr.ndim == 3:
                return arr.astype(np.float32)
    raise ValueError(f"Could not find HSI data in {path}")


def bicubic_upsample(lr_hsi, scale_factor):
    """Upsample LR-HSI using bicubic interpolation."""
    h, w = lr_hsi.shape[0] * scale_factor, lr_hsi.shape[1] * scale_factor
    bands = lr_hsi.shape[2]
    out = []
    for b in range(bands):
        out.append(cv2.resize(lr_hsi[:, :, b], (w, h), interpolation=cv2.INTER_CUBIC))
    return np.stack(out, axis=2)


def to_torch_chw(arr_hwc, device):
    """Convert HWC numpy array to CHW torch tensor with normalization."""
    arr01 = normalize01(arr_hwc).astype(np.float32)
    return torch.from_numpy(arr01).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser(description="HSISR test on Kaggle HSI+RGB data")
    ap.add_argument("hsi_dir", help="Directory with HSI .mat files")
    ap.add_argument("--rgb_dir", type=str, required=True, help="Directory with RGB .mat files")
    ap.add_argument("--weights", type=str, required=True, help="Path to .pth weights")
    ap.add_argument("--sf", type=int, default=4, help="Scale factor (4/8/16)")
    ap.add_argument("--cuda", type=int, default=1, help="Use CUDA if available")
    ap.add_argument("--limit", type=int, default=0, help="Limit images (0=all)")
    
    # Model hyperparams (must match weights)
    ap.add_argument("--n_feats", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--n_subs", type=int, default=8)
    ap.add_argument("--n_ovls", type=int, default=2)
    ap.add_argument("--n_colors", type=int, default=31)
    ap.add_argument("--n_scale", type=int, default=None, help="Scale for model (defaults to sf)")

    args = ap.parse_args()

    device = torch.device("cuda" if (args.cuda == 1 and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")
    
    # Default n_scale to sf if not provided
    if args.n_scale is None:
        args.n_scale = args.sf

    # Load model
    model = DeepShare(
        n_subs=args.n_subs,
        n_ovls=args.n_ovls,
        n_feats=args.n_feats,
        n_blocks=args.n_blocks,
        n_colors=args.n_colors,
        n_scale=args.n_scale,
        res_scale=0.1,
    )
    
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"Loaded weights from: {args.weights}")

    # Find HSI files
    hsi_mats = sorted(glob.glob(os.path.join(args.hsi_dir, "*.mat")))
    if not hsi_mats:
        raise SystemExit(f"No .mat files found in {args.hsi_dir}")

    if args.limit and args.limit > 0:
        hsi_mats = hsi_mats[: args.limit]

    print(f"Testing on {len(hsi_mats)} images (SF={args.sf})\n")

    metrics_list = []

    with torch.no_grad():
        for i, hsi_path in enumerate(hsi_mats):
            fname = os.path.basename(hsi_path)
            
            # Load GT HSI
            gt_hsi = load_mat(hsi_path)
            
            # Load RGB (try corresponding file in rgb_dir)
            rgb_stem = os.path.splitext(fname)[0]
            rgb_path = os.path.join(args.rgb_dir, f"{rgb_stem}.mat")
            if not os.path.exists(rgb_path):
                # Fallback: synthesize RGB from HSI bands
                idx_b, idx_g, idx_r = 7, 15, 23
                msi = np.stack([gt_hsi[..., idx_b], gt_hsi[..., idx_g], gt_hsi[..., idx_r]], axis=-1)
                print(f"{fname}: RGB synthesized (not found in rgb_dir)")
            else:
                msi = load_mat(rgb_path)
                print(f"{fname}: RGB loaded", end="")

            # Create LR-HSI by downsampling GT
            h, w = gt_hsi.shape[0] // args.sf, gt_hsi.shape[1] // args.sf
            lr_hsi = cv2.resize(gt_hsi, (w, h), interpolation=cv2.INTER_AREA)

            # Bicubic interpolation for reference
            lms = bicubic_upsample(lr_hsi, args.sf)

            # HSISR expects: LR-HSI (upsampled) + HR-MSI as input
            # Prepare inputs
            lms_torch = to_torch_chw(lms, device)
            msi_torch = to_torch_chw(msi, device)

            # Inference
            with torch.no_grad():
                pred_hsi = model(lms_torch, msi_torch, modality="spectral")
            
            pred_hsi = pred_hsi.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_hsi = normalize01(pred_hsi).astype(np.float32)

            # Compute metrics
            gt_norm = normalize01(gt_hsi).astype(np.float32)
            
            psnr = compute_metrics(pred_hsi, gt_norm)['PSNR']
            sam = compute_metrics(pred_hsi, gt_norm)['SAM']
            ergas = compute_metrics(pred_hsi, gt_norm)['ERGAS']
            ssim = compute_metrics(pred_hsi, gt_norm)['SSIM']

            metrics_list.append({'PSNR': psnr, 'SAM': sam, 'ERGAS': ergas, 'SSIM': ssim})
            
            print(f" → PSNR={psnr:.2f}, SAM={sam:.2f}, ERGAS={ergas:.2f}, SSIM={ssim:.4f}")

    # Summary
    print("\n" + "=" * 70)
    avg_psnr = np.mean([m['PSNR'] for m in metrics_list])
    avg_sam = np.mean([m['SAM'] for m in metrics_list])
    avg_ergas = np.mean([m['ERGAS'] for m in metrics_list])
    avg_ssim = np.mean([m['SSIM'] for m in metrics_list])
    
    print(f"Average (SF={args.sf}):")
    print(f"  PSNR:  {avg_psnr:.2f} dB")
    print(f"  SAM:   {avg_sam:.2f}°")
    print(f"  ERGAS: {avg_ergas:.2f}")
    print(f"  SSIM:  {avg_ssim:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
