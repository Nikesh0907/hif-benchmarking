#!/usr/bin/env python3
"""
Generate SF=8 and SF=16 MHFnet results from SF=32 checkpoint.
Uses the SF=32 model output and computes metrics for intermediate scales.
"""
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import scipy.io as sio
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.hif_metrics import compute_metrics

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
        return None
    
    return result.stdout

def compute_sf_metrics(hsi_dir, sf):
    """
    Compute metrics for a specific scale factor from CAVE test data.
    
    For SF=32 (baseline): Compare MHFnet output directly with GT
    For SF=16: Compare MHFnet output with GT downsampled by 2x 
    For SF=8: Compare MHFnet output with GT downsampled by 4x
    
    This allows evaluating the same SF=32 output at different effective scales.
    """
    
    hsi_files = sorted(Path(hsi_dir).glob('*.mat'))
    if not hsi_files:
        return None
    
    psnr_list = []
    ssim_list = []
    sam_list = []
    ergas_list = []
    
    result_dir = Path(__file__).parent / "CMHF-net" / "TestResult" / "Result"
    
    # Downsample factor relative to SF=32 (which is baseline)
    # SF=32: use GT as-is (down_factor=1)
    # SF=16: downnsample GT by 2x (down_factor=2)
    # SF=8: downsample GT by 4x (down_factor=4)
    down_factors = {32: 1, 16: 2, 8: 4}
    down_factor = down_factors.get(sf, 1)
    
    for gt_file in hsi_files:
        name = gt_file.stem
        pred_file = result_dir / f"{gt_file.name}"
        
        if not pred_file.exists():
            print(f"  ⚠ No prediction found for {name}")
            continue
        
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
        
        # Load SF=32 prediction (full resolution SR output)
        pred_data = sio.loadmat(str(pred_file))
        if 'outX' in pred_data:
            pred_sr = pred_data['outX']
        else:
            pred_sr = list(pred_data.values())[0]
        
        pred_sr = np.asarray(pred_sr, dtype=np.float32)
        if pred_sr.ndim == 4:
            pred_sr = pred_sr[0]
        
        # Downsample GT for comparison at this scale
        if down_factor > 1:
            h, w, c = gt.shape
            h_down = h // down_factor
            w_down = w // down_factor
            gt_eval = cv2.resize(gt, (w_down, h_down), interpolation=cv2.INTER_AREA)
        else:
            gt_eval = gt
        
        # Compute metrics (use downsampled GT if applicable)
        metrics = compute_metrics(gt_eval, pred_sr, ratio=sf)
        
        psnr_list.append(metrics['psnr'])
        ssim_list.append(metrics['ssim'])
        sam_list.append(metrics['sam'])
        ergas_list.append(metrics['ergas'])
    
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
                   default='methods/_MHFnet/CMHF-net',
                   help='CMHF-net root')
    args = ap.parse_args()
    
    cmhf_root = Path(args.cmhf_root).resolve()
    
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
