#!/usr/bin/env python3
"""
Test TSFN directly on Kaggle CAVE .mat files (NO .tif intermediate).

This script reads HSI/RGB .mat directly, applies degradation,
and runs TSFN inference without any preprocessing pipeline issues.

Usage:
    python methods/_TSFN/test_kaggle_direct.py \
        --hsi_dir /kaggle/input/cave-dataset-2/Data/Test/HSI \
        --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
        --sf 8
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import glob
from pathlib import Path
from PIL import Image

# Import TSFN modules
TSFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(TSFN_ROOT))

import torch
import torch.nn as nn
import time

# Add repo root for metrics
REPO_ROOT = TSFN_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.hif_metrics import compute_metrics, normalize01

from model import Net
from dataset import get_lrhsi


def load_mat_kaggle(path):
    """Load Kaggle CAVE .mat file (auto-detect key)."""
    mat_data = sio.loadmat(path)
    
    # Try common keys used in Kaggle CAVE
    for key in ['hsi', 'gt', 'msi', 'X', 'ref']:
        if key in mat_data:
            arr = mat_data[key]
            break
    else:
        # Fallback: use first non-meta key
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        arr = mat_data[keys[0]] if keys else None
    
    if arr is None:
        raise ValueError(f"No suitable key found in {path}")
    
    arr = np.asarray(arr, dtype=np.float32)
    
    # Normalize to [0, 1]
    arr_max = float(np.nanmax(arr)) if arr.size else 1.0
    if arr_max > 1.0:
        arr = arr / arr_max
    
    return np.clip(arr, 0.0, 1.0)  # Ensure [0, 1]


def main():
    parser = argparse.ArgumentParser(description='Test TSFN on Kaggle CAVE (direct .mat loading)')
    parser.add_argument('--hsi_dir', required=True, help='Kaggle HSI .mat directory')
    parser.add_argument('--rgb_dir', required=True, help='Kaggle RGB .mat directory')
    parser.add_argument('--sf', type=int, default=8, choices=[8, 16, 32], 
                       help='Scale factor (8, 16, or 32)')
    parser.add_argument('--weights', type=str, 
                       default='methods/_TSFN/models/ssfsr_9layers_epoch500.pkl',
                       help='Path to TSFN weights .pkl')
    parser.add_argument('--cuda', type=int, default=1, help='Use CUDA if available')
    args = parser.parse_args()
    
    # Map SF to degradation_mode
    degradation_mode = {8: 0, 16: 2, 32: 3}.get(args.sf, 0)
    
    # Step 1: Load model
    print("=" * 70)
    print(f"[1/2] Loading TSFN model...")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(args.weights, map_location=device)
    model = Net(HSI_num_residuals=6, RGB_num_residuals=6)
    
    # Handle DataParallel checkpoint
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"✓ Model loaded from {args.weights}")
    print(f"✓ Device: {device}")
    
    # Step 2: Load test data and run inference
    print(f"\n[2/2] Running inference (SF={args.sf}, degradation_mode={degradation_mode})...")
    
    hsi_files = sorted(glob.glob(os.path.join(args.hsi_dir, '*.mat')))
    if not hsi_files:
        print(f"ERROR: No .mat files found in {args.hsi_dir}")
        return
    
    results = {'psnr': [], 'ssim': [], 'sam': [], 'ergas': []}
    
    with torch.no_grad():
        for idx, hsi_path in enumerate(hsi_files):
            name = os.path.basename(hsi_path).replace('.mat', '')
            
            # Load HSI and RGB
            hsi_full = load_mat_kaggle(hsi_path)  # (H, W, 31) in [0, 1]
            if hsi_full.ndim == 4:
                hsi_full = hsi_full[0]
            if hsi_full.shape[2] > 31:
                hsi_full = hsi_full[:, :, :31]
            
            # Load RGB
            rgb_path = os.path.join(args.rgb_dir, os.path.basename(hsi_path))
            if os.path.exists(rgb_path):
                rgb_full = load_mat_kaggle(rgb_path)  # (H, W, 3) in [0, 1]
                if rgb_full.ndim == 4:
                    rgb_full = rgb_full[0]
                if rgb_full.shape[2] > 3:
                    rgb_full = rgb_full[:, :, :3]
            else:
                # Synthesize from HSI bands
                idx_r, idx_g, idx_b = 23, 15, 7
                if hsi_full.shape[2] >= idx_r + 1:
                    rgb_full = np.stack([hsi_full[..., idx_r], hsi_full[..., idx_g], hsi_full[..., idx_b]], axis=-1)
                else:
                    rgb_full = np.tile(hsi_full[..., :1], (1, 1, 3))
            
            # Convert to CHW format as required by TSFN model
            hsi_chw = hsi_full.transpose(2, 0, 1).astype(np.float32)  # (31, H, W) [0, 1]
            rgb_chw = rgb_full.transpose(2, 0, 1).astype(np.float32)  # (3, H, W) [0, 1]
            
            # Apply degradation to HSI to get LR-HSI
            lr_hsi_chw = get_lrhsi(hsi_chw, degradation_mode).astype(np.float32)  # (31, H/sf, W/sf) [0, 1]-ish
            
            # Normalize to [0, 1] after degradation (important!)
            lr_hsi_chw = np.clip(lr_hsi_chw / max(lr_hsi_chw.max(), 1.0), 0.0, 1.0)
            
            # Convert to tensors
            lr_hsi_tensor = torch.from_numpy(lr_hsi_chw).unsqueeze(0).to(device)  # (1, 31, H/sf, W/sf)
            rgb_tensor = torch.from_numpy(rgb_chw).unsqueeze(0).to(device)        # (1, 3, H, W)
            hsi_tensor = torch.from_numpy(hsi_chw).unsqueeze(0).to(device)        # (1, 31, H, W)
            
            start = time.time()
            pred = model(lr_hsi_tensor, rgb_tensor)  # (1, 31, H, W) in [0, 1]
            elapsed = time.time() - start
            
            # Extract predictions and convert to numpy
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 31) [0, 1]
            gt_np = hsi_chw.transpose(1, 2, 0)  # (H, W, 31) [0, 1]
            
            # Clip to [0, 1]
            pred_np = np.clip(pred_np, 0.0, 1.0)
            gt_np = np.clip(gt_np, 0.0, 1.0)
            
            # Debug output on first image
            if idx == 0:
                print(f"\n  DEBUG: Data ranges")
                print(f"    Pred: min={pred_np.min():.6f}, max={pred_np.max():.6f}, mean={pred_np.mean():.6f}")
                print(f"    GT:   min={gt_np.min():.6f}, max={gt_np.max():.6f}, mean={gt_np.mean():.6f}\n")
            
            # Compute metrics
            m = compute_metrics(gt_np, pred_np, ratio=args.sf)
            for k in results:
                results[k].append(m[k])
            
            print(f"  {idx+1}/{len(hsi_files)}: {name}: PSNR={m['psnr']:.2f} SAM={m['sam']:.2f}° ERGAS={m['ergas']:.3f} SSIM={m['ssim']:.4f} ({elapsed:.2f}s)")
    
    # Summary
    print("\n" + "=" * 70)
    avg = {k: np.mean(v) for k, v in results.items()}
    print(f"AVERAGE (SF={args.sf}, n={len(hsi_files)}):")
    print(f"  PSNR: {avg['psnr']:.2f}")
    print(f"  SAM:  {avg['sam']:.2f}°")
    print(f"  ERGAS: {avg['ergas']:.3f}")
    print(f"  SSIM: {avg['ssim']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
