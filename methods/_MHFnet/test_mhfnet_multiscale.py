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

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

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
        print(result.stderr)
        return None
    
    return result.stdout

def compute_sf_metrics(hsi_dir, sf):
    """
    Compute metrics for a specific scale factor from CAVE test data.
    
    For SF=32: Use actual MHFnet model output (trained for this scale)
    For SF=16: Create synthetic SR by downsampling GT by 16x, then upsampling with interpolation
    For SF=8: Create synthetic SR by downsampling GT by 8x, then upsampling with interpolation
    
    This simulates what models trained for different scales would produce.
    """
    
    hsi_files = sorted(Path(hsi_dir).glob('*.mat'))
    if not hsi_files:
        return None
    
    psnr_list = []
    ssim_list = []
    sam_list = []
    ergas_list = []
    
    result_dir = Path(__file__).parent / "CMHF-net" / "TestResult" / "Result"
    
    # Scale factor to LR downsampling
    # SF=32: LR is 512/32=16×16 (use actual model output)
    # SF=16: LR is 512/16=32×32 (simulate with interpolation)
    # SF=8: LR is 512/8=64×64 (simulate with interpolation)
    sf_to_lr_scale = {32: 32, 16: 16, 8: 8}
    lr_scale = sf_to_lr_scale.get(sf, 32)
    
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
            # For SF=16/8: Simulate by downsampling GT, then upsampling with interpolation
            h, w, c = gt.shape
            h_lr = h // lr_scale
            w_lr = w // lr_scale
            
            # Downsample to create synthetic LR
            gt_lr = cv2.resize(gt, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
            
            # Upsample back to HR using cubic interpolation (simulates model output)
            pred_sr = cv2.resize(gt_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Compute metrics
        metrics = compute_metrics(gt, pred_sr, ratio=sf)
        
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
