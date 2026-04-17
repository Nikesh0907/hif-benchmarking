#!/usr/bin/env python3
"""
Test TSFN on Kaggle CAVE dataset - clean wrapper.

TSFN expects .tif stacks (31 HSI bands + 3 RGB) but we have separate .mat files.
This script:
1. Converts Kaggle HSI/RGB .mat files to TSFN format .tif stacks
2. Runs TSFN inference
3. Computes metrics (PSNR, SAM, ERGAS, SSIM)

Usage:
    python methods/_TSFN/test_kaggle_cave.py \
        --hsi_dir /kaggle/input/cave-dataset-2/Data/Test/HSI \
        --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
        --sf 8

Degradation modes:
    SF=8: degradation_mode=0 or 1
    SF=16: degradation_mode=2
    SF=32: degradation_mode=3
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import glob
from pathlib import Path

# Import TSFN test module
TSFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(TSFN_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import tifffile as tiff

# Add repo root for metrics
REPO_ROOT = TSFN_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.hif_metrics import compute_metrics, normalize01

from model import Net
from dataset import get_lrhsi


def load_mat_auto(path):
    """Load .mat file, auto-detect key for HSI/RGB data."""
    mat_data = sio.loadmat(path)
    # Try common keys
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
    if arr.max() > 1.0:
        arr = arr / 255.0 if arr.max() <= 255 else arr / arr.max()
    return np.clip(arr, 0, 1)


def prepare_tsfn_data(hsi_dir, rgb_dir, tsfn_data_dir):
    """Prepare TSFN test data by creating stacked .tif files.
    
    TSFN format: .tif file with shape (34, H, W) where:
    - Bands 0-30: HSI (31 bands)
    - Bands 31-33: RGB (3 bands)
    """
    os.makedirs(tsfn_data_dir, exist_ok=True)
    
    hsi_files = sorted(glob.glob(os.path.join(hsi_dir, '*.mat')))
    
    for hsi_path in hsi_files:
        name = os.path.basename(hsi_path).replace('.mat', '.tif')
        
        # Load HSI
        hsi = load_mat_auto(hsi_path)
        if hsi.ndim == 4:
            hsi = hsi[0]
        if hsi.shape[2] > 31:
            hsi = hsi[:, :, :31]
        
        # Load RGB or synthesize
        rgb_path = os.path.join(rgb_dir, os.path.basename(hsi_path))
        if os.path.exists(rgb_path):
            rgb = load_mat_auto(rgb_path)
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.shape[2] > 3:
                rgb = rgb[:, :, :3]
            print(f"  {name}: RGB loaded")
        else:
            # Synthesize RGB from HSI bands
            idx_r, idx_g, idx_b = 23, 15, 7
            if hsi.shape[2] >= idx_r + 1:
                rgb = np.stack([hsi[..., idx_r], hsi[..., idx_g], hsi[..., idx_b]], axis=-1)
            else:
                rgb = np.tile(hsi[..., :1], (1, 1, 3))
            print(f"  {name}: RGB synthesized")
        
        # Stack HSI (CHW) + RGB (CHW) and save as .tif
        # TSFN expects shape (34, H, W) where bands 0-30=HSI, 31-33=RGB
        hsi_chw = hsi.transpose(2, 0, 1)  # HWC -> CHW
        rgb_chw = rgb.transpose(2, 0, 1)  # HWC -> CHW
        stacked = np.vstack([hsi_chw, rgb_chw]).astype(np.float32)  # (34, H, W)
        
        out_path = os.path.join(tsfn_data_dir, name)
        tiff.imwrite(out_path, (stacked * 255).astype(np.uint8))
    
    print(f"Prepared {len(hsi_files)} TSFN test images in {tsfn_data_dir}")


class TsfnTestDataset(Dataset):
    """TSFN test dataset from .tif stacks."""
    
    def __init__(self, tif_dir, degradation_mode):
        self.tif_list = sorted(glob.glob(os.path.join(tif_dir, '*.tif')))
        self.degradation_mode = degradation_mode
    
    def __getitem__(self, idx):
        """Load stacked .tif and return (lr_hsi, rgb, hr_hsi) as tensors."""
        img = tiff.imread(self.tif_list[idx])  # (34, H, W) uint8
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        hsi = img[:31, :, :]  # (31, H, W)
        rgb = img[31:34, :, :]  # (3, H, W)
        
        # Create LR-HSI via degradation
        lr_hsi = get_lrhsi(hsi, self.degradation_mode)  # (31, H/sf, W/sf)
        
        # Convert to tensors (model expects CHW)
        hsi_tensor = torch.from_numpy(hsi).float()
        rgb_tensor = torch.from_numpy(rgb).float()
        lr_hsi_tensor = torch.from_numpy(lr_hsi).float()
        
        return lr_hsi_tensor, rgb_tensor, hsi_tensor
    
    def __len__(self):
        return len(self.tif_list)


def main():
    parser = argparse.ArgumentParser(description='Test TSFN on Kaggle CAVE')
    parser.add_argument('--hsi_dir', required=True, help='Kaggle HSI .mat directory')
    parser.add_argument('--rgb_dir', required=True, help='Kaggle RGB .mat directory')
    parser.add_argument('--sf', type=int, default=8, choices=[8, 16, 32], 
                       help='Scale factor (8, 16, or 32)')
    parser.add_argument('--weights', type=str, 
                       default='methods/_TSFN/models/ssfsr_9layers_epoch500.pkl',
                       help='Path to TSFN weights .pkl')
    parser.add_argument('--cuda', type=int, default=1, help='Use CUDA if available')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    
    # Map SF to degradation mode
    if args.sf == 8:
        degradation_mode = 0
    elif args.sf == 16:
        degradation_mode = 2
    else:  # 32
        degradation_mode = 3
    
    print("=" * 70)
    print(f"TSFN Test on CAVE (SF={args.sf})")
    print("=" * 70)
    
    # Step 1: Prepare TSFN data from Kaggle format
    print("\n[1/3] Preparing TSFN test data...")
    temp_data_dir = './temp_tsfn_cave'
    prepare_tsfn_data(args.hsi_dir, args.rgb_dir, temp_data_dir)
    
    # Step 2: Load model
    print("\n[2/3] Loading TSFN model...")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(args.weights, map_location=device)
    model = Net(HSI_num_residuals=6, RGB_num_residuals=6)
    
    # Handle DataParallel checkpoint (remove 'module.' prefix if present)
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"✓ Model loaded from {args.weights}")
    
    # Step 3: Run inference and compute metrics
    print(f"\n[3/3] Running inference (degradation_mode={degradation_mode})...")
    dataset = TsfnTestDataset(temp_data_dir, degradation_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    results = {'psnr': [], 'ssim': [], 'sam': [], 'ergas': []}
    
    with torch.no_grad():
        for i, (lr_hsi, rgb, hsi_gt) in enumerate(dataloader):
            lr_hsi = lr_hsi.to(device)
            rgb = rgb.to(device)
            hsi_gt = hsi_gt.to(device)
            
            start = time.time()
            pred = model(lr_hsi, rgb)  # Output: (B, 31, H, W)
            elapsed = time.time() - start
            
            # Convert to numpy for metric computation
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            gt_np = hsi_gt[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            
            # Debug: Print value ranges (only on first iteration)
            if i == 0:
                print(f"\n  DEBUG: Data ranges before normalize01()")
                print(f"    Pred: min={pred_np.min():.6f}, max={pred_np.max():.6f}, mean={pred_np.mean():.6f}")
                print(f"    GT:   min={gt_np.min():.6f}, max={gt_np.max():.6f}, mean={gt_np.mean():.6f}")
            
            # Normalize
            pred_norm = normalize01(pred_np).astype(np.float32)
            gt_norm = normalize01(gt_np).astype(np.float32)
            
            # Ensure both are in [0, 1] and at same scale
            # (GT may be clipped at <1.0 due to input data range, so clip both to [0,1])
            pred_norm = np.clip(pred_norm, 0.0, 1.0)
            gt_norm = np.clip(gt_norm, 0.0, 1.0)
            
            if i == 0:
                print(f"  DEBUG: Data ranges after normalize01() + clipping")
                print(f"    Pred: min={pred_norm.min():.6f}, max={pred_norm.max():.6f}, mean={pred_norm.mean():.6f}")
                print(f"    GT:   min={gt_norm.min():.6f}, max={gt_norm.max():.6f}, mean={gt_norm.mean():.6f}\n")
            
            # Compute metrics
            m = compute_metrics(gt_norm, pred_norm, ratio=args.sf)
            for k in results:
                results[k].append(m[k])
            
            print(f"  {i+1}/{len(dataset)}: PSNR={m['psnr']:.2f} SAM={m['sam']:.2f}° ERGAS={m['ergas']:.3f} SSIM={m['ssim']:.4f} ({elapsed:.2f}s)")
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    avg = {k: np.mean(v) for k, v in results.items()}
    print(f"AVERAGE (SF={args.sf}, n={len(dataset)}):")
    print(f"  PSNR: {avg['psnr']:.2f}")
    print(f"  SAM:  {avg['sam']:.2f}°")
    print(f"  ERGAS: {avg['ergas']:.3f}")
    print(f"  SSIM: {avg['ssim']:.4f}")
    print("=" * 70)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_data_dir, ignore_errors=True)
    print(f"\n✓ Test complete. Temp data cleaned up.")


if __name__ == '__main__':
    main()
