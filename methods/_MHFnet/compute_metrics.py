#!/usr/bin/env python3
"""
Compute PSNR/SSIM/SAM/ERGAS metrics for MHFnet on CAVE dataset.
Same metrics format as DBIN.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from tools.hif_metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser(description="Compute MHFnet CAVE metrics (PSNR/SSIM/SAM/ERGAS)")
    ap.add_argument('--pred_dir', required=True, help='Predictions directory (MHFnet TestResult/Result)')
    ap.add_argument('--gt_dir', required=True, help='Ground truth directory (CAVEdata/X)')
    ap.add_argument('--scale_factor', type=int, default=32, help='Scale factor for ERGAS')
    ap.add_argument('--split_list', default='', help='Optional split list .mat file')
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists():
        print(f"❌ Pred dir not found: {pred_dir}")
        return 1

    if not gt_dir.exists():
        print(f"❌ GT dir not found: {gt_dir}")
        return 1

    # Load predictions
    pred_files = sorted(pred_dir.glob('*.mat'))
    if not pred_files:
        print(f"❌ No .mat files in {pred_dir}")
        return 1

    print("="*70)
    print("MHFnet Metrics (PSNR/SSIM/SAM/ERGAS)")
    print("="*70)
    print(f"Predictions: {pred_dir}")
    print(f"Ground truth: {gt_dir}")
    print()

    all_metrics = []
    header_printed = False

    for pred_file in pred_files:
        name = pred_file.stem.replace('.mat', '')
        gt_file = gt_dir / f"{pred_file.name}"

        if not gt_file.exists():
            print(f"⚠ GT not found for {name}, skipping")
            continue

        try:
            # Load prediction
            pred_data = sio.loadmat(str(pred_file))
            if 'outX' in pred_data:
                pred = pred_data['outX']
            elif 'sr_hsi' in pred_data:
                pred = pred_data['sr_hsi']
            else:
                pred = list(pred_data.values())[0]

            # Load ground truth
            gt_data = sio.loadmat(str(gt_file))
            if 'msi' in gt_data:
                gt = gt_data['msi']
            elif 'img' in gt_data:
                gt = gt_data['img']
            elif 'hsi' in gt_data:
                gt = gt_data['hsi']
            else:
                gt = list(gt_data.values())[0]

            # Squeeze batch dimension if present
            if pred.ndim == 4:
                pred = pred[0]
            if gt.ndim == 4:
                gt = gt[0]

            # Compute metrics
            metrics = compute_metrics(gt, pred, ratio=args.scale_factor)

            # Print header once
            if not header_printed:
                print("name,psnr,ssim,sam,ergas")
                header_printed = True

            # Print result
            print(f"{name},{metrics['psnr']:.4f},{metrics['ssim']:.4f},{metrics['sam']:.4f},{metrics['ergas']:.4f}")
            all_metrics.append(metrics)

        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            continue

    if not all_metrics:
        print("❌ No metrics computed")
        return 1

    # Compute averages
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_sam = np.mean([m['sam'] for m in all_metrics])
    avg_ergas = np.mean([m['ergas'] for m in all_metrics])

    print(f"avg,{avg_psnr:.4f},{avg_ssim:.4f},{avg_sam:.4f},{avg_ergas:.4f}")
    print("="*70)
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
