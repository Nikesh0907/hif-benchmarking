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


# Import DBIN's proper metric functions (more reliable than sewar)
def compute_sam(image1, image2):
    """Compute SAM in degrees between spectral vectors."""
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]
    
    # Expects HWC format
    if image1.ndim == 3 and image1.shape[0] < image1.shape[2]:  # CHW format
        image1 = image1.transpose(1, 2, 0)
    if image2.ndim == 3 and image2.shape[0] < image2.shape[2]:  # CHW format
        image2 = image2.transpose(1, 2, 0)
    
    h, w, c = image1.shape
    image1 = np.reshape(image1, (h * w, c))
    image2 = np.reshape(image2, (h * w, c))
    mole = np.sum(np.multiply(image1, image2), axis=1)
    image1_norm = np.sqrt(np.sum(np.square(image1), axis=1))
    image2_norm = np.sqrt(np.sum(np.square(image2), axis=1))
    deno = np.multiply(image1_norm, image2_norm)
    sam = np.rad2deg(np.arccos((mole + 1e-11) / (deno + 1e-11)))
    return np.mean(sam)


def compute_ergas(mse, out, sf=8):
    """Compute ERGAS metric from MSE map and output."""
    out = np.asarray(out)
    if out.ndim == 4:
        out = out[0]
    
    # Expects HWC format
    if out.ndim == 3 and out.shape[0] < out.shape[2]:  # CHW format
        out = out.transpose(1, 2, 0)
    
    h, w, c = out.shape
    out = np.reshape(out, (h * w, c))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (c, 1))
    out_mean = np.reshape(out_mean, (c, 1))
    ergas = 100.0 / float(sf) * np.sqrt(np.mean(mse / (out_mean ** 2 + 1e-12)))
    return ergas


def compute_psnr(image1, image2, data_range=1.0):
    """Compute PSNR."""
    image1 = np.asarray(image1, dtype=np.float32)
    image2 = np.asarray(image2, dtype=np.float32)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return psnr


def compute_ssim(image1, image2, data_range=1.0, win_size=11):
    """Compute SSIM per-band and average."""
    from skimage.metrics import structural_similarity
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]
    
    # Expects HWC format
    if image1.ndim == 3 and image1.shape[0] < image1.shape[2]:  # CHW format
        image1 = image1.transpose(1, 2, 0)
    if image2.ndim == 3 and image2.shape[0] < image2.shape[2]:  # CHW format
        image2 = image2.transpose(1, 2, 0)
    
    h, w, c = image1.shape
    ssim_total = 0.0
    for i in range(c):
        ssim_total += structural_similarity(image1[:, :, i], image2[:, :, i], data_range=data_range)
    return ssim_total / c


def load_mat_kaggle(path):
    """Load Kaggle CAVE .mat or .tif file (auto-detect format and orientation)."""
    if path.endswith('.tif'):
        # Load .tif file directly
        import tifffile as tiff
        img = tiff.imread(path).astype(np.float32)
        
        # Auto-detect format: could be (34, H, W) or (H, W, 34) or (H, W, 31)
        if img.ndim == 3:
            # Identify which dimension is the spectral dimension
            if img.shape[0] in [31, 34]:  # First dim is bands
                img = img.transpose(1, 2, 0)  # Convert (B, H, W) -> (H, W, B)
        
        img_max = img.max()
        if img_max > 1.0:
            img = img / img_max
        return np.clip(img, 0.0, 1.0)
    
    # Load .mat file
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
    
    # Auto-detect orientation: (B, H, W) or (H, W, B)
    if arr.ndim == 3 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = arr.transpose(1, 2, 0)  # (B, H, W) -> (H, W, B)
    
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
    
    # Fallback: check for subdirectories or .tif files
    if not hsi_files:
        print(f"⚠️  No .mat files in {args.hsi_dir}, checking subdirectories...")
        hsi_files = sorted(glob.glob(os.path.join(args.hsi_dir, '*/*.mat')))
    
    if not hsi_files:
        print(f"⚠️  Still no .mat files, checking for .tif files...")
        hsi_files = sorted(glob.glob(os.path.join(args.hsi_dir, '*.tif')))
        hsi_files += sorted(glob.glob(os.path.join(args.hsi_dir, '*/*.tif')))
    
    if not hsi_files:
        print(f"\n❌ ERROR: No data files (.mat or .tif) found in {args.hsi_dir}")
        print(f"Directory contents:")
        import subprocess
        subprocess.run(['ls', '-lh', args.hsi_dir], capture_output=False)
        return
    
    results = {'psnr': [], 'ssim': [], 'sam': [], 'ergas': []}
    
    with torch.no_grad():
        for idx, hsi_path in enumerate(hsi_files):
            name = os.path.basename(hsi_path).replace('.mat', '')
            
            # Load HSI and RGB
            hsi_full = load_mat_kaggle(hsi_path)  # (H, W, 31) in [0, 1]
            
            # CRITICAL VALIDATION on first iteration
            if idx == 0:
                print(f"\n⚠️  DATA VALIDATION:")
                print(f"    HSI file '{name}' shape: {hsi_full.shape}")
                if hsi_full.ndim >= 3 and hsi_full.shape[2] == 3:
                    print(f"    ❌ ERROR: Loaded RGB (3 channels) instead of HSI (31 channels)!")
                    print(f"    ❌ Are your HSI_DIR and RGB_DIR arguments SWAPPED?")
                    return
                elif hsi_full.ndim >= 3 and hsi_full.shape[2] == 31:
                    print(f"    ✅ OK: Loaded HSI with 31 channels")
                else:
                    print(f"    ⚠️  WARNING: Got {hsi_full.shape[2] if hsi_full.ndim >= 3 else '?'} channels (expected 31)")
            
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
                print(f"\n  DEBUG: Data ranges (HWC format)")
                print(f"    Pred: shape={pred_np.shape}, min={pred_np.min():.6f}, max={pred_np.max():.6f}")
                print(f"    GT:   shape={gt_np.shape}, min={gt_np.min():.6f}, max={gt_np.max():.6f}\n")
            
            # Compute metrics using DBIN-style functions (proven to work correctly)
            psnr = compute_psnr(gt_np, pred_np, data_range=1.0)
            sam = compute_sam(gt_np, pred_np)
            ssim = compute_ssim(gt_np, pred_np, data_range=1.0)
            
            # ERGAS needs MSE map per-channel
            mse_per_channel = np.mean((gt_np - pred_np) ** 2, axis=(0, 1))  # (C,)
            ergas = compute_ergas(mse_per_channel, pred_np, sf=args.sf)
            
            m = {'psnr': psnr, 'sam': sam, 'ergas': ergas, 'ssim': ssim}
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
