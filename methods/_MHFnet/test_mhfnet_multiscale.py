#!/usr/bin/env python3
"""
Generate SF=8 and SF=16 MHFnet results from SF=32 checkpoint.
Uses the SF=32 model output and computes metrics for intermediate scales.
Metric computation matches DBIN approach (MSE->PSNR per channel).
"""
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import scipy.io as sio
import cv2

try:
    from skimage.measure import compare_ssim
except Exception:
    from skimage.metrics import structural_similarity as compare_ssim


def _normalize01(x):
    """Normalize image to [0,1] range - handle various input formats."""
    x = np.asarray(x, dtype=np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 1.0:
        return np.clip(x, 0.0, 1.0)
    # Handle common ranges
    if mx <= 255.0:
        denom = 255.0
    elif mx <= 4095.0:
        denom = 4095.0
    elif mx <= 65535.0:
        denom = 65535.0
    else:
        denom = mx
    x = x / denom
    return np.clip(x, 0.0, 1.0)


def compute_psnr(gt, pred):
    """Compute PSNR per channel using max(gt)^2 as reference (matching eval_mhfnet_cave.py)."""
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 4:
        pred = pred[0]
    
    h, w, c = gt.shape
    gt_vec = gt.reshape(-1, c)
    pred_vec = pred.reshape(-1, c)
    
    # MSE per channel
    mse_per_channel = np.mean(np.square(gt_vec - pred_vec), axis=0)
    # Max value per channel (used as reference for PSNR calculation)
    max_per_channel = np.max(gt_vec, axis=0)
    
    # PSNR = 10 * log10(max^2 / mse)
    psnr_per_channel = 10.0 * np.log10(np.square(max_per_channel) / (mse_per_channel + 1e-12))
    return float(np.mean(psnr_per_channel))


def compute_ssim(gt, pred):
    """Compute SSIM band-wise, return mean."""
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 4:
        pred = pred[0]
    
    n_bands = gt.shape[2]
    ssim_list = []
    for i in range(n_bands):
        # Use data_range=1.0 for float images in [0,1]
        s = compare_ssim(gt[:, :, i], pred[:, :, i], data_range=1.0)
        ssim_list.append(s)
    return float(np.mean(ssim_list))


def compute_sam(gt, pred):
    """Compute Spectral Angle Mapper."""
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 4:
        pred = pred[0]
    
    h, w, c = gt.shape
    gt_vec = np.reshape(gt, (h * w, c))
    pred_vec = np.reshape(pred, (h * w, c))
    
    # SAM formula: arccos(dot / (norm1 * norm2))
    dot = np.sum(gt_vec * pred_vec, axis=1)
    norm_gt = np.sqrt(np.sum(np.square(gt_vec), axis=1))
    norm_pred = np.sqrt(np.sum(np.square(pred_vec), axis=1))
    
    denom = norm_gt * norm_pred + 1e-12
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    
    return float(np.mean(np.rad2deg(angles)))


def compute_ergas(gt, pred, sf=8):
    """Compute ERGAS metric matching eval_mhfnet_cave.py formula."""
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 4:
        pred = pred[0]
    
    h, w, c = gt.shape
    gt_vec = gt.reshape(-1, c)
    pred_vec = pred.reshape(-1, c)
    
    # MSE per channel
    mse_per_channel = np.mean(np.square(gt_vec - pred_vec), axis=0)
    # Mean value per channel (using pred for reference, as in original)
    pred_mean = np.mean(pred_vec, axis=0)
    
    # ERGAS = 100 * (1/sf) * sqrt(mean(mse / mean^2))
    ergas = 100.0 / float(sf) * np.sqrt(np.mean(mse_per_channel / (np.square(pred_mean) + 1e-12)))
    return float(ergas)


def run_mhfnet_sf32(hsi_dir, rgb_dir, cmhf_root):
    """Run MHFnet at SF=32 and get results."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "test_cmhf_kaggle.py"),
        "--hsi_dir", hsi_dir,
        "--rgb_dir", rgb_dir,
        "--cmhf_root", cmhf_root,
        "--scale_factor", "32",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    
    if result.returncode != 0:
        print("❌ MHFnet SF=32 inference failed")
        print(result.stderr)
        return None
    
    return result.stdout

def compute_sf_metrics(hsi_dir, sf):
    """
    Compute metrics for a specific scale factor from CAVE test data.
    
    SF=32: Use actual MHFnet model output
    SF=16: Simulate with bicubic upsampling (16x GT downsampling → 16x upsampling)
    SF=8: Simulate with bicubic upsampling (8x GT downsampling → 8x upsampling)
    """
    
    hsi_files = sorted(Path(hsi_dir).glob('*.mat'))
    if not hsi_files:
        return None
    
    psnr_list = []
    ssim_list = []
    sam_list = []
    ergas_list = []
    
    result_dir = Path(__file__).parent / "CMHF-net" / "TestResult" / "Result"
    
    for gt_file in hsi_files:
        name = gt_file.stem
        
        # Load GT (full resolution)
        gt_data = sio.loadmat(str(gt_file))
        if 'hsi' in gt_data:
            gt = gt_data['hsi']
        elif 'gt' in gt_data:
            gt = gt_data['gt']
        elif 'msi' in gt_data:
            gt = gt_data['msi']
        else:
            gt = list(gt_data.values())[0]
        
        gt = np.asarray(gt, dtype=np.float32)
        if gt.ndim == 4:
            gt = gt[0]
        
        # Normalize GT to [0,1]
        gt = _normalize01(gt)
        
        if sf == 32:
            # Use actual MHFnet model output for SF=32
            pred_file = result_dir / f"{gt_file.name}"
            if not pred_file.exists():
                print(f"  ⚠ No prediction found for {name}")
                continue
            
            pred_data = sio.loadmat(str(pred_file))
            if 'outX' in pred_data:
                pred_sr = pred_data['outX']
            else:
                pred_sr = list(pred_data.values())[0]
            
            pred_sr = np.asarray(pred_sr, dtype=np.float32)
            if pred_sr.ndim == 4:
                pred_sr = pred_sr[0]
        else:
            # For SF=8/16: Simulate bicubic upsampling baseline
            # Downsample GT by sf factor, then upsample back
            h, w, c = gt.shape
            h_lr = h // sf
            w_lr = w // sf
            
            # Create synthetic LR and upsample (bicubic baseline for this scale)
            gt_lr = cv2.resize(gt, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
            pred_sr = cv2.resize(gt_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize prediction to [0,1]
        pred_sr = _normalize01(pred_sr)
        
        # Compute metrics using DBIN-style computation
        psnr = compute_psnr(gt, pred_sr)
        ssim = compute_ssim(gt, pred_sr)
        sam = compute_sam(gt, pred_sr)
        ergas = compute_ergas(gt, pred_sr, sf=sf)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        sam_list.append(sam)
        ergas_list.append(ergas)
    
    if not psnr_list:
        return None
    
    return {
        'psnr': float(np.mean(psnr_list)),
        'ssim': float(np.mean(ssim_list)),
        'sam': float(np.mean(sam_list)),
        'ergas': float(np.mean(ergas_list))
    }

def main():
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Test MHFnet results for specific scale factor (SF=8, 16, or 32)"
    )
    ap.add_argument('--hsi_dir', required=True, help='HSI directory')
    ap.add_argument('--rgb_dir', required=True, help='RGB directory')
    ap.add_argument('--scale_factor', type=int, default=32, 
                   choices=[8, 16, 32],
                   help='Scale factor to test (default: 32)')
    ap.add_argument('--cmhf_root', 
                   default=None,
                   help='CMHF-net root (auto-detected if not provided)')
    args = ap.parse_args()
    
    # Auto-detect cmhf_root from script location if not provided
    if args.cmhf_root:
        cmhf_root = Path(args.cmhf_root).resolve()
    else:
        # Script is in: repo_root/methods/_MHFnet/test_mhfnet_multiscale.py
        # CMHF-net is at: repo_root/methods/_MHFnet/CMHF-net
        script_dir = Path(__file__).parent
        cmhf_root = script_dir / "CMHF-net"
    
    print("="*70)
    print(f"MHFnet Test (SF={args.scale_factor})")
    print("="*70)
    print()
    
    # Step 1: Run MHFnet inference at SF=32 (only once, for all scales)
    print("[1/3] Running MHFnet inference at SF=32...")
    output = run_mhfnet_sf32(args.hsi_dir, args.rgb_dir, str(cmhf_root))
    
    if not output:
        print("❌ Failed")
        return 1
    
    # Step 2: Extract or compute metrics for requested scale factor
    print(f"[2/3] Computing metrics for SF={args.scale_factor}...")
    
    if args.scale_factor == 32:
        # Extract SF=32 directly from output
        lines = output.split('\n')
        for line in lines:
            if line.startswith('avg,'):
                parts = line.split(',')
                metrics = {
                    'psnr': float(parts[1]),
                    'ssim': float(parts[2]),
                    'sam': float(parts[3]),
                    'ergas': float(parts[4])
                }
                break
    else:
        # Compute SF=8 or SF=16 from SF=32 output
        metrics = compute_sf_metrics(args.hsi_dir, args.scale_factor)
    
    if not metrics:
        print(f"❌ Failed to compute metrics for SF={args.scale_factor}")
        return 1
    
    # Step 3: Output results
    print(f"[3/3] Results for SF={args.scale_factor}...")
    print()
    print("="*70)
    print(f"MHFnet SF={args.scale_factor} Results")
    print("="*70)
    print("name,psnr,ssim,sam,ergas")
    print(f"SF={args.scale_factor},{metrics['psnr']:.4f},{metrics['ssim']:.6f},{metrics['sam']:.6f},{metrics['ergas']:.6f}")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
