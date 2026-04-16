#!/usr/bin/env python3
"""
Test CMHF-net on Kaggle CAVE dataset.

This script:
1. Prepares CAVE data from Kaggle format into CMHF-net format
2. Runs CMHF-net inference via CAVEmain.py (testAll mode)
3. Computes metrics using eval_mhfnet_cave.py

Usage:
    python methods/_MHFnet/test_cmhf_kaggle.py \
        --hsi_dir /kaggle/input/cave-dataset-2/Data/Test/HSI \
        --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB

Requirements:
    - TensorFlow 1.x (tf.compat.v1)
    - scipy, numpy, scikit-image
"""

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np
import scipy.io as sio
from glob import glob

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.hif_metrics import normalize01


def prepare_cmhf_data(hsi_dir, rgb_dir, cmhf_root):
    """Prepare CAVE data in CMHF-net format.
    
    Creates:
    - CAVEdata/X/<name>.mat with key 'msi' (H,W,31) [0,1]
    - CAVEdata/Y/<name>.mat with key 'RGB' (H,W,3) [0,1]
    - CAVEdata/Z/<name>.mat with key 'Zmsi' (H/32,W/32,31) [0,1]
    """
    import cv2
    
    cave_root = os.path.join(cmhf_root, 'CAVEdata')
    x_dir = os.path.join(cave_root, 'X')
    y_dir = os.path.join(cave_root, 'Y')
    z_dir = os.path.join(cave_root, 'Z')
    
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)
    os.makedirs(z_dir, exist_ok=True)
    
    # Load response matrix for RGB synthesis if needed
    response_path = os.path.join(cave_root, 'response coefficient.mat')
    response = None
    if os.path.exists(response_path):
        try:
            response = sio.loadmat(response_path).get('R', None)
        except:
            response = None
    
    hsi_files = sorted(glob(os.path.join(hsi_dir, '*.mat')))
    
    for hsi_path in hsi_files:
        name = os.path.basename(hsi_path)
        
        # Load HSI
        hsi_data = sio.loadmat(hsi_path)
        if 'hsi' in hsi_data:
            hsi = hsi_data['hsi']
        elif 'gt' in hsi_data:
            hsi = hsi_data['gt']
        else:
            raise KeyError(f"No 'hsi'/'gt' key in {hsi_path}")
        
        hsi = normalize01(np.asarray(hsi, dtype=np.float32))
        if hsi.ndim == 4:
            hsi = hsi[0]
        if hsi.shape[2] > 31:
            hsi = hsi[:, :, :31]
        
        # Load or synthesize RGB
        rgb_path = os.path.join(rgb_dir, name)
        rgb = None
        if os.path.exists(rgb_path):
            try:
                rgb_data = sio.loadmat(rgb_path)
                if 'rgb' in rgb_data:
                    rgb = rgb_data['rgb']
                elif 'msi' in rgb_data:
                    rgb = rgb_data['msi']
                if rgb is not None:
                    rgb = normalize01(np.asarray(rgb, dtype=np.float32))
                    if rgb.ndim == 4:
                        rgb = rgb[0]
                    if rgb.shape[2] > 3:
                        rgb = rgb[:, :, :3]
            except:
                rgb = None
        
        # If RGB not found, synthesize from response matrix or bands
        if rgb is None and response is not None:
            try:
                # RGB = HSI @ R^T (response matrix multiplication)
                response_use = np.asarray(response)
                if response_use.shape[0] == 3:
                    response_use = response_use.T  # (3, 31) -> (31, 3)
                h, w = hsi.shape[:2]
                hsi_reshaped = hsi.reshape(-1, 31)  # (H*W, 31)
                rgb = hsi_reshaped @ response_use  # (H*W, 3)
                rgb = rgb.reshape(h, w, 3)
                rgb = normalize01(rgb.astype(np.float32))
                print(f"  {name}: RGB synthesized (response matrix)")
            except:
                rgb = None
        
        if rgb is None:
            # Fallback: use simple band selection
            idx_r, idx_g, idx_b = 23, 15, 7
            if hsi.shape[2] >= idx_r + 1:
                rgb = np.stack([hsi[..., idx_r], hsi[..., idx_g], hsi[..., idx_b]], axis=-1)
            else:
                rgb = np.tile(hsi[..., :1], (1, 1, 3))
            print(f"  {name}: RGB from band selection")
        else:
            print(f"  {name}: RGB loaded")
        
        # Save X (full-res HSI)
        sio.savemat(os.path.join(x_dir, name), {'msi': hsi.astype(np.float32)})
        
        # Save Y (RGB)
        sio.savemat(os.path.join(y_dir, name), {'RGB': rgb.astype(np.float32)})
        
        # Save Z (downsampled HSI, 1/32 scale)
        h_z, w_z = hsi.shape[0] // 32, hsi.shape[1] // 32
        hsi_z = cv2.resize(hsi, (w_z, h_z), interpolation=cv2.INTER_AREA)
        sio.savemat(os.path.join(z_dir, name), {'Zmsi': hsi_z.astype(np.float32)})
    
    print(f"Prepared {len(hsi_files)} images in CMHF-net format")


def run_cmhf_inference_via_main(cmhf_root):
    """Run CMHF-net inference using CAVEmain.py testAll mode."""
    import subprocess
    import tempfile
    
    cmhf_dir = os.path.abspath(cmhf_root)
    print(f"Running CMHF-net via CAVEmain.py (testAll mode)...")
    
    # Read CAVEmain.py and modify FLAGS.mode to 'testAll'
    cavemain_path = os.path.join(cmhf_dir, 'CAVEmain.py')
    with open(cavemain_path, 'r') as f:
        content = f.read()
    
    # Replace mode flag from 'test' to 'testAll'
    modified_content = content.replace("FLAGS.mode, 'test'", "FLAGS.mode, 'testAll'")
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=cmhf_dir) as tmp:
        tmp.write(modified_content)
        tmp_path = tmp.name
    
    try:
        # Run the modified script
        result = subprocess.run(
            [sys.executable, tmp_path],
            cwd=cmhf_dir,
            timeout=600,
            capture_output=False
        )
        if result.returncode == 0:
            print("✓ Inference completed")
        else:
            print(f"⚠ CAVEmain.py returned code {result.returncode}")
        return os.path.join(cmhf_root, 'TestResult', 'Result')
    except subprocess.TimeoutExpired:
        print("⚠ Inference timed out after 10 minutes")
        return os.path.join(cmhf_root, 'TestResult', 'Result')
    except Exception as e:
        print(f"⚠ Inference failed: {e}")
        return os.path.join(cmhf_root, 'TestResult', 'Result')
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except:
            pass


def main():
    ap = argparse.ArgumentParser(description="Test CMHF-net on Kaggle CAVE dataset")
    ap.add_argument('--hsi_dir', required=True, help='Kaggle HSI directory')
    ap.add_argument('--rgb_dir', required=True, help='Kaggle RGB directory')
    ap.add_argument('--cmhf_root', default='methods/_MHFnet/CMHF-net', help='CMHF-net root')
    ap.add_argument('--skip_prep', action='store_true', help='Skip data preparation')
    ap.add_argument('--skip_inference', action='store_true', help='Skip inference, only compute metrics')
    args = ap.parse_args()
    
    cmhf_root = os.path.abspath(args.cmhf_root)
    
    print("=" * 70)
    print("CMHF-net Test on CAVE Dataset")
    print("=" * 70)
    
    # Step 1: Prepare data
    if not args.skip_prep:
        print("\n[1/3] Preparing CAVE data in CMHF-net format...")
        prepare_cmhf_data(args.hsi_dir, args.rgb_dir, cmhf_root)
    
    # Step 2: Run inference
    result_dir = os.path.join(cmhf_root, 'TestResult', 'Result')
    if not args.skip_inference:
        print("\n[2/3] Running CMHF-net inference...")
        result_dir = run_cmhf_inference_via_main(cmhf_root) or result_dir
    
    # Step 3: Compute metrics using eval_mhfnet_cave.py
    print("\n[3/3] Computing metrics...")
    eval_script = os.path.join(os.path.dirname(__file__), 'eval_mhfnet_cave.py')
    
    cmd = [
        sys.executable, eval_script,
        '--pred_dir', result_dir,
        '--gt_dir', os.path.join(cmhf_root, 'CAVEdata', 'X'),
        '--split_list', os.path.join(cmhf_root, 'CAVEdata', 'List'),
        '--sf', '32'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error computing metrics: {e}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
