#!/usr/bin/env python3
"""
Test CMHF-net on Kaggle CAVE dataset.

This script:
1. Prepares CAVE data from Kaggle format into CMHF-net format
2. Runs CMHF-net inference (testAll)
3. Computes metrics (PSNR, SAM, ERGAS, SSIM)

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
        response = sio.loadmat(response_path).get('R', None)
    
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
        if os.path.exists(rgb_path):
            rgb_data = sio.loadmat(rgb_path)
            if 'rgb' in rgb_data:
                rgb = rgb_data['rgb']
            elif 'msi' in rgb_data:
                rgb = rgb_data['msi']
            else:
                rgb = None
            if rgb is not None:
                rgb = normalize01(np.asarray(rgb, dtype=np.float32))
        else:
            rgb = None
        
        # If RGB not found, synthesize from HSI
        if rgb is None and response is not None:
            # RGB = HSI @ R^T (response matrix multiplication)
            # response shape: (31, 3) or (3, 31)
            response_use = response
            if response_use.shape[0] == 3:
                response_use = response_use.T  # (3, 31) -> (31, 3)
            h, w = hsi.shape[:2]
            hsi_reshaped = hsi.reshape(-1, 31)  # (H*W, 31)
            rgb = hsi_reshaped @ response_use  # (H*W, 3)
            rgb = rgb.reshape(h, w, 3)
            rgb = normalize01(rgb.astype(np.float32))
            print(f"  {name}: RGB synthesized from response matrix")
        elif rgb is None:
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


def run_cmhf_inference(cmhf_root):
    """Run CMHF-net inference on prepared CAVE data."""
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    # Change to CMHF-net directory and import modules
    cmhf_py = os.path.join(cmhf_root)
    sys.path.insert(0, cmhf_py)
    
    import CAVEmain
    
    # Create result directory
    result_dir = os.path.join(cmhf_root, 'TestResult', 'Result')
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Running CMHF-net inference...")
    # The CAVEmain will handle inference via testAll mode
    # This is a bit tricky since CAVEmain uses tf.app.flags
    # We'll do a simple manual inference instead
    
    import CAVE_dataReader as Crd
    import MHFnet
    import MyLib as ML
    
    outDim = 31
    upRank = 12
    
    # Load the trained model
    model_path = os.path.join(cmhf_root, 'temp', 'TrainedNet')
    
    # Get test image names
    cave_data_root = os.path.join(cmhf_root, 'CAVEdata')
    x_dir = os.path.join(cave_data_root, 'X')
    test_names = sorted([f for f in os.listdir(x_dir) if f.endswith('.mat')])
    
    # Create TF session and run inference
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        # Define model inputs
        y_input = tf.placeholder(tf.float32, shape=[None, None, 3], name='y_input')
        Z_input = tf.placeholder(tf.float32, shape=[None, None, outDim], name='Z_input')
        
        # Build network
        net = MHFnet.net_tf(y_input, Z_input, upRank)
        
        # Restore weights
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_path, 'model-epoch-30'))
        print(f"Loaded model from {model_path}/model-epoch-30")
        
        # Run inference on each test image
        for name in test_names:
            x_path = os.path.join(x_dir, name)
            y_path = os.path.join(cave_data_root, 'Y', name)
            z_path = os.path.join(cave_data_root, 'Z', name)
            
            # Load data
            x = sio.loadmat(x_path)['msi']  # Ground truth
            y = sio.loadmat(y_path)['RGB']  # RGB input
            z = sio.loadmat(z_path)['Zmsi']  # Downsampled HSI
            
            # Run inference
            out = sess.run(net, feed_dict={y_input: y, Z_input: z})
            
            # Save result
            out_path = os.path.join(result_dir, name)
            sio.savemat(out_path, {'outX': out.astype(np.float32)})
            
            print(f"  {name}: saved")
    
    print(f"Inference complete. Results in {result_dir}")
    return result_dir


def compute_metrics(cmhf_root, pred_dir, sf=32):
    """Compute PSNR, SAM, ERGAS, SSIM on test results."""
    from skimage.metrics import structural_similarity as compare_ssim
    
    cave_root = os.path.join(cmhf_root, 'CAVEdata')
    x_dir = os.path.join(cave_root, 'X')
    test_names = sorted([f for f in os.listdir(x_dir) if f.endswith('.mat')])
    
    results = []
    
    for name in test_names:
        x_path = os.path.join(x_dir, name)
        pred_path = os.path.join(pred_dir, name)
        
        if not os.path.exists(pred_path):
            print(f"  {name}: prediction not found, skipping")
            continue
        
        # Load GT and prediction
        gt = sio.loadmat(x_path)['msi']  # (H, W, 31)
        pred = sio.loadmat(pred_path)['outX']  # (H, W, 31) or (1, H, W, 31)
        
        if pred.ndim == 4:
            pred = pred[0]
        
        # Compute metrics
        # PSNR
        mse = np.mean((gt - pred) ** 2)
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else 100.0
        
        # SAM (degrees)
        gt_flat = gt.reshape(-1, 31)
        pred_flat = pred.reshape(-1, 31)
        dots = np.sum(gt_flat * pred_flat, axis=1)
        norms_gt = np.linalg.norm(gt_flat, axis=1)
        norms_pred = np.linalg.norm(pred_flat, axis=1)
        norms = norms_gt * norms_pred + 1e-10
        sam = np.mean(np.arccos(np.clip(dots / norms, -1, 1))) * (180 / np.pi)
        
        # SSIM (band-wise average)
        ssim = 0.0
        for b in range(31):
            ssim += float(compare_ssim(gt[..., b], pred[..., b], data_range=1.0))
        ssim /= 31.0
        
        # ERGAS
        mse_per_band = np.mean((gt - pred) ** 2, axis=(0, 1))
        pred_mean = np.mean(pred.reshape(-1, 31), axis=0)
        ergas = 100.0 / sf * np.sqrt(np.mean(mse_per_band / (pred_mean ** 2 + 1e-12)))
        
        results.append({'name': name, 'psnr': psnr, 'sam': sam, 'ergas': ergas, 'ssim': ssim})
        print(f"  {name}: PSNR={psnr:.2f} SAM={sam:.2f} ERGAS={ergas:.4f} SSIM={ssim:.4f}")
    
    # Average
    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_sam = np.mean([r['sam'] for r in results])
        avg_ergas = np.mean([r['ergas'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        print(f"\nAverage: PSNR={avg_psnr:.2f} SAM={avg_sam:.2f} ERGAS={avg_ergas:.4f} SSIM={avg_ssim:.4f}")
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Test CMHF-net on Kaggle CAVE dataset")
    ap.add_argument('--hsi_dir', required=True, help='Kaggle HSI directory')
    ap.add_argument('--rgb_dir', required=True, help='Kaggle RGB directory')
    ap.add_argument('--cmhf_root', default='methods/_MHFnet/CMHF-net', help='CMHF-net root')
    ap.add_argument('--skip_prep', action='store_true', help='Skip data preparation')
    ap.add_argument('--skip_inference', action='store_true', help='Skip inference, only compute metrics')
    args = ap.parse_args()
    
    print("=" * 70)
    print("CMHF-net Test on CAVE Dataset")
    print("=" * 70)
    
    # Step 1: Prepare data
    if not args.skip_prep:
        print("\n[1/3] Preparing CAVE data in CMHF-net format...")
        prepare_cmhf_data(args.hsi_dir, args.rgb_dir, args.cmhf_root)
    
    # Step 2: Run inference
    result_dir = os.path.join(args.cmhf_root, 'TestResult', 'Result')
    if not args.skip_inference:
        print("\n[2/3] Running CMHF-net inference...")
        try:
            result_dir = run_cmhf_inference(args.cmhf_root)
        except Exception as e:
            print(f"Inference failed: {e}")
            print(f"Trying to use existing results from {result_dir}...")
    
    # Step 3: Compute metrics
    print("\n[3/3] Computing metrics...")
    compute_metrics(args.cmhf_root, result_dir, sf=32)
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
