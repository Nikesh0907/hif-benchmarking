#!/usr/bin/env python3
"""
HSRnet Test - DBIN-style output format
Model expects: rgbNet(ms_lr, rgb_hr) where:
  - ms_lr: LR HSI (H/sf, W/sf, 31)
  - rgb_hr: HR RGB/MSI (H, W, 3)
Output: predicted HR HSI (H, W, 31)
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import scipy.io
from PIL import Image
import tensorflow as tf

# Add repo root to path (works from anywhere)
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from methods._HSRnet.HSRnet import rgbNet
from tools.hif_metrics import compute_metrics, normalize01


def resize_bicubic(arr: np.ndarray, out_hw: tuple) -> np.ndarray:
    """Resize to (oh, ow) using PIL bicubic"""
    oh, ow = out_hw
    out = []
    for b in range(arr.shape[2]):
        band = (arr[:, :, b] * 255).astype(np.uint8)
        band_img = Image.fromarray(band)
        band_resized = band_img.resize((ow, oh), Image.BICUBIC)
        out.append(np.array(band_resized) / 255.0)
    return np.stack(out, axis=2)


def downsample_hsi(hsi: np.ndarray, sf: int) -> np.ndarray:
    """Downsample HSI by scale factor using area resampling"""
    h, w = hsi.shape[:2]
    return resize_bicubic(hsi, (h // sf, w // sf))


def main():
    parser = argparse.ArgumentParser(description="HSRnet Test (DBIN-style output)")
    parser.add_argument('--hsi_dir', required=True, help='Directory with GT HSI .mat files')
    parser.add_argument('--rgb_dir', required=True, help='Directory with RGB/MSI .mat files')
    parser.add_argument('--model_path', required=True, help='Path to pretrained checkpoint dir')
    parser.add_argument('--sf', type=int, default=4, help='Scale factor')
    parser.add_argument('--num_images', type=int, default=12, help='Number of test images')
    parser.add_argument('--crop', type=int, default=0, help='Center crop size (0=no crop)')
    
    args = parser.parse_args()

    if not os.path.exists(args.hsi_dir):
        print(f"❌ HSI dir not found: {args.hsi_dir}")
        return 1
    if not os.path.exists(args.rgb_dir):
        print(f"❌ RGB dir not found: {args.rgb_dir}")
        return 1
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        return 1

    # Disable eager execution for TF2
    if tf.__version__.startswith("2"):
        tf.compat.v1.disable_eager_execution()

    print("="*80)
    print(f"HSRnet Test (SF={args.sf})")
    print("="*80)

    # Get test images
    hsi_files = sorted(glob.glob(os.path.join(args.hsi_dir, '*.mat')))[:args.num_images]
    print(f"Found {len(hsi_files)} HSI files\n")

    if not hsi_files:
        print("❌ No .mat files found!")
        return 1

    # Build TensorFlow graph
    sz = args.crop if args.crop > 0 else 512
    
    r_hp = tf.compat.v1.placeholder(shape=[1, sz, sz, 3], dtype=tf.float32)
    m_hp = tf.compat.v1.placeholder(shape=[1, sz // args.sf, sz // args.sf, 31], dtype=tf.float32)
    lms_p = tf.compat.v1.placeholder(shape=[1, sz, sz, 31], dtype=tf.float32)

    # Forward pass: output = network(ms, rgb) + lms
    rs = rgbNet(m_hp, r_hp, reuse=False)
    out = tf.clip_by_value(tf.add(rs, lms_p), 0.0, 1.0)

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    results = []
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Load checkpoint
        ckpt = tf.train.latest_checkpoint(args.model_path)
        if not ckpt:
            print(f"❌ No checkpoint found in {args.model_path}")
            return 1
        
        saver.restore(sess, ckpt)
        print(f"✓ Model loaded from {ckpt}\n")
        print("[2/2] Running inference (SF={})...".format(args.sf))

        for idx, hsi_path in enumerate(hsi_files, start=1):
            name = Path(hsi_path).stem
            
            try:
                # Load GT HSI
                hsi_mat = scipy.io.loadmat(hsi_path)
                if 'hsi' in hsi_mat:
                    gt_hsi = hsi_mat['hsi'].astype(np.float32)
                elif 'ref' in hsi_mat:
                    gt_hsi = hsi_mat['ref'].astype(np.float32)
                else:
                    print(f"  ⚠ {idx}/{len(hsi_files)}: {name}: No 'hsi'/'ref' key - skipping")
                    continue

                # Load RGB/MSI
                rgb_path = os.path.join(args.rgb_dir, f'{name}.mat')
                if os.path.exists(rgb_path):
                    rgb_mat = scipy.io.loadmat(rgb_path)
                    if 'msi' in rgb_mat:
                        rgb = rgb_mat['msi'].astype(np.float32)
                    elif 'ms' in rgb_mat:
                        rgb = rgb_mat['ms'].astype(np.float32)
                    else:
                        print(f"  ⚠ {idx}/{len(hsi_files)}: {name}: No RGB key - skipping")
                        continue
                else:
                    print(f"  ⚠ {idx}/{len(hsi_files)}: {name}: No RGB file - skipping")
                    continue

                # Normalize
                gt_hsi = normalize01(gt_hsi)
                rgb = normalize01(rgb)
                
                # Make sure RGB is 3 channels
                if rgb.ndim == 3 and rgb.shape[2] > 3:
                    rgb = rgb[:, :, :3]

                # Optional center crop
                if args.crop > 0:
                    h, w = gt_hsi.shape[:2]
                    if h >= args.crop and w >= args.crop:
                        y0 = (h - args.crop) // 2
                        x0 = (w - args.crop) // 2
                        gt_hsi = gt_hsi[y0:y0+args.crop, x0:x0+args.crop]
                        rgb = rgb[y0:y0+args.crop, x0:x0+args.crop]

                # Downsample GT to create LR-HSI
                lr_hsi = downsample_hsi(gt_hsi, args.sf)
                
                # Upsample LR-HSI using bicubic (for residual connection)
                lr_hsi_up = resize_bicubic(lr_hsi, (gt_hsi.shape[0], gt_hsi.shape[1]))

                # Prepare batches [1, H, W, C]
                lr_batch = np.expand_dims(lr_hsi, 0)
                rgb_batch = np.expand_dims(rgb, 0)
                lr_up_batch = np.expand_dims(lr_hsi_up, 0)

                # Run inference
                pred = sess.run(out, feed_dict={
                    m_hp: lr_batch,
                    r_hp: rgb_batch,
                    lms_p: lr_up_batch
                })
                pred = pred[0]

                # Compute metrics
                metrics = compute_metrics(gt_hsi, pred, sf=args.sf)
                
                results.append({
                    'name': name,
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'sam': metrics['sam'],
                    'ergas': metrics['ergas']
                })
                
                print(f"  {idx:2d}/{len(hsi_files)}: {name:30s}: "
                      f"PSNR={metrics['psnr']:6.2f} SAM={metrics['sam']:6.2f}° "
                      f"ERGAS={metrics['ergas']:8.3f} SSIM={metrics['ssim']:.4f}")

            except Exception as e:
                print(f"  ⚠ {idx}/{len(hsi_files)}: {name}: {str(e)[:50]}")

    # Summary
    if results:
        print("\n" + "="*80)
        psnrs = [r['psnr'] for r in results]
        ssims = [r['ssim'] for r in results]
        sams = [r['sam'] for r in results]
        ergass = [r['ergas'] for r in results]
        
        print(f"AVERAGE (SF={args.sf}, n={len(results)}):")
        print(f"  PSNR: {np.mean(psnrs):.2f}")
        print(f"  SSIM: {np.mean(ssims):.4f}")
        print(f"  SAM:  {np.mean(sams):.2f}°")
        print(f"  ERGAS: {np.mean(ergass):.2f}")
        print("="*80)
        return 0
    else:
        print("\n❌ No valid results!")
        return 1


if __name__ == '__main__':
    exit(main())
