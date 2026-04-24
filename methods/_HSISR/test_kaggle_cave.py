#!/usr/bin/env python3
"""
Test HSISR on Kaggle CAVE format directly (no .mat conversion needed).

HSISR model forward signature: model(x_lr, lms_upsampled, modality="spectral")
where x_lr is LR-HSI and lms_upsampled is bicubic-upsampled MS.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import glob
from pathlib import Path

# Import HSISR model
HSISR_ROOT = Path(__file__).parent
sys.path.insert(0, str(HSISR_ROOT))
sys.path.insert(0, str(HSISR_ROOT.parent.parent))

import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter
from PIL import Image
import time

from BlockModule import DeepShare
from basicModule import default_conv
from tools.hif_metrics import normalize01


# ============================================================================
# Metric Functions (use DBIN-style for accuracy)
# ============================================================================
def compute_sam(image1, image2):
    """Compute SAM in degrees."""
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]
    
    if image1.ndim == 3 and image1.shape[0] < image1.shape[2]:
        image1 = image1.transpose(1, 2, 0)
    if image2.ndim == 3 and image2.shape[0] < image2.shape[2]:
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
    """Compute ERGAS metric from MSE per-channel."""
    out = np.asarray(out)
    if out.ndim == 4:
        out = out[0]
    
    if out.ndim == 3 and out.shape[0] < out.shape[2]:
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


def compute_ssim(image1, image2, data_range=1.0):
    """Compute SSIM per-band and average."""
    from skimage.metrics import structural_similarity
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]
    
    if image1.ndim == 3 and image1.shape[0] < image1.shape[2]:
        image1 = image1.transpose(1, 2, 0)
    if image2.ndim == 3 and image2.shape[0] < image2.shape[2]:
        image2 = image2.transpose(1, 2, 0)
    
    h, w, c = image1.shape
    ssim_total = 0.0
    for i in range(c):
        ssim_total += structural_similarity(image1[:, :, i], image2[:, :, i], data_range=data_range)
    return ssim_total / c


# ============================================================================
# Data Loading
# ============================================================================
def load_mat_kaggle(path):
    """Load Kaggle CAVE .mat (auto-detect format)."""
    mat_data = sio.loadmat(path)
    
    for key in ['hsi', 'gt', 'msi', 'X']:
        if key in mat_data:
            arr = mat_data[key]
            break
    else:
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        arr = mat_data[keys[0]] if keys else None
    
    if arr is None:
        raise ValueError(f"No suitable key in {path}")
    
    arr = np.asarray(arr, dtype=np.float32)
    
    # Auto-detect CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = arr.transpose(1, 2, 0)
    
    # Normalize to [0, 1]
    arr_max = float(np.nanmax(arr)) if arr.size else 1.0
    if arr_max > 1.0:
        arr = arr / arr_max
    
    return np.clip(arr, 0.0, 1.0)


def bicubic_upsample(img_lr, sf):
    """Bicubic upsample using PIL."""
    h, w, c = img_lr.shape
    h_hr, w_hr = h * sf, w * sf
    
    img_up = np.zeros((h_hr, w_hr, c), dtype=np.float32)
    for i in range(c):
        pil_img = Image.fromarray((img_lr[:, :, i] * 255).astype(np.uint8))
        pil_up = pil_img.resize((w_hr, h_hr), Image.BICUBIC)
        img_up[:, :, i] = np.array(pil_up, dtype=np.float32) / 255.0
    
    return img_up


def bicubic_downsample(img, sf):
    """Bicubic downsample using PIL (matches training!)."""
    h, w, c = img.shape
    h_lr, w_lr = h // sf, w // sf
    
    img_lr = np.zeros((h_lr, w_lr, c), dtype=np.float32)
    for i in range(c):
        pil_img = Image.fromarray((img[:, :, i] * 255).astype(np.uint8))
        pil_ds = pil_img.resize((w_lr, h_lr), Image.BICUBIC)
        img_lr[:, :, i] = np.array(pil_ds, dtype=np.float32) / 255.0
    
    return img_lr


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Test HSISR on Kaggle CAVE')
    parser.add_argument('--hsi_dir', required=True, help='HSI .mat directory')
    parser.add_argument('--msi_dir', required=True, help='MSI/RGB .mat directory')
    parser.add_argument('--sf', type=int, default=4, help='Scale factor')
    parser.add_argument('--weights', type=str, 
                       default='methods/_HSISR/models/Cave_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Wed_Mar_31_03:00:46_2021.pth',
                       help='Path to HSISR weights')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--num_images', type=int, default=12)
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"HSISR Test on Kaggle CAVE (SF={args.sf})")
    print("=" * 70)
    
    # Load model
    print(f"\n[1/2] Loading HSISR model...")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    model = DeepShare(
        n_subs=8, n_ovls=2, n_colors=31, n_blocks=3, n_feats=256,
        n_scale=args.sf, res_scale=0.1, use_share=True, conv=default_conv
    )
    
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"✓ Model loaded from {args.weights}")
    print(f"✓ Device: {device}")
    
    # Get test files
    hsi_files = sorted(glob.glob(os.path.join(args.hsi_dir, '*.mat')))[:args.num_images]
    
    if not hsi_files:
        print(f"ERROR: No .mat files found in {args.hsi_dir}")
        return
    
    # Run inference
    print(f"\n[2/2] Running inference (SF={args.sf})...")
    results = {'psnr': [], 'ssim': [], 'sam': [], 'ergas': []}
    
    with torch.no_grad():
        for idx, hsi_path in enumerate(hsi_files):
            name = os.path.basename(hsi_path).replace('.mat', '')
            
            # Load HSI
            hsi_full = load_mat_kaggle(hsi_path)
            if hsi_full.ndim == 4:
                hsi_full = hsi_full[0]
            if hsi_full.shape[2] > 31:
                hsi_full = hsi_full[:, :, :31]
            
            # Load MSI (RGB)
            msi_path = os.path.join(args.msi_dir, os.path.basename(hsi_path))
            if os.path.exists(msi_path):
                msi_full = load_mat_kaggle(msi_path)
                if msi_full.ndim == 4:
                    msi_full = msi_full[0]
                if msi_full.shape[2] > 3:
                    msi_full = msi_full[:, :, :3]
            else:
                # Synthesize from HSI bands
                idx_r, idx_g, idx_b = 23, 15, 7
                msi_full = np.stack([hsi_full[..., idx_r], hsi_full[..., idx_g], hsi_full[..., idx_b]], axis=-1)
            
            # Ensure divisible by SF
            h, w = hsi_full.shape[:2]
            h_new = (h // args.sf) * args.sf
            w_new = (w // args.sf) * args.sf
            hsi_full = hsi_full[:h_new, :w_new, :]
            
            # Degrade HSI using BICUBIC (same as training!)
            lr_hsi = bicubic_downsample(hsi_full, args.sf)
            
            # Bicubic upsample LR-HSI
            lms_full = bicubic_upsample(lr_hsi, args.sf)
            
            # Convert to tensors (CHW, [0,1])
            lr_hsi_norm = normalize01(lr_hsi).astype(np.float32)
            lms_norm = normalize01(lms_full).astype(np.float32)
            
            lr_hsi_t = torch.from_numpy(lr_hsi_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            lms_t = torch.from_numpy(lms_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Forward
            start = time.time()
            pred = model(lr_hsi_t, lms_t, modality="spectral")
            elapsed = time.time() - start
            
            # Convert output to numpy HWC (model output is [0,1])
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
            gt_np = normalize01(hsi_full).astype(np.float32)  # Ensure same normalization!
            
            # Clip and compute metrics
            pred_np = np.clip(pred_np, 0.0, 1.0)
            gt_np = np.clip(gt_np, 0.0, 1.0)
            
            psnr = compute_psnr(gt_np, pred_np, data_range=1.0)
            sam = compute_sam(gt_np, pred_np)
            ssim = compute_ssim(gt_np, pred_np, data_range=1.0)
            mse_per_channel = np.mean((gt_np - pred_np) ** 2, axis=(0, 1))
            ergas = compute_ergas(mse_per_channel, pred_np, sf=args.sf)
            
            results['psnr'].append(psnr)
            results['sam'].append(sam)
            results['ergas'].append(ergas)
            results['ssim'].append(ssim)
            
            print(f"  {idx+1}/{len(hsi_files)}: {name}: PSNR={psnr:.2f} SAM={sam:.2f}° ERGAS={ergas:.3f} SSIM={ssim:.4f} ({elapsed:.2f}s)")
    
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
