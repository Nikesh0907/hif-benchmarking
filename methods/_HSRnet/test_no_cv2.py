#!/usr/bin/env python3
"""Test HSRnet on CAVE/Harvard - NO CV2 DEPENDENCY (uses PIL/scipy instead)"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import scipy.io
from PIL import Image
import tensorflow as tf

from methods._HSRnet.HSRnet import rgbNet
from tools.hif_metrics import compute_metrics, normalize01


def _center_crop(arr: np.ndarray, crop_hw: Tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[0], arr.shape[1]
    ch, cw = crop_hw
    if h < ch or w < cw:
        return arr
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return arr[y0:y0+ch, x0:x0+cw, ...]


def _resize_hsi(hsi: np.ndarray, out_hw: Tuple[int, int], *, interpolation='bicubic') -> np.ndarray:
    """Resize HSI using PIL (no cv2 needed)"""
    oh, ow = out_hw
    out = []
    
    if interpolation == 'bicubic':
        pil_interp = Image.BICUBIC
    elif interpolation == 'area':
        pil_interp = Image.LANCZOS
    else:
        pil_interp = Image.BILINEAR
    
    for b in range(hsi.shape[2]):
        band = (hsi[:, :, b] * 255).astype(np.uint8)
        band_img = Image.fromarray(band)
        band_resized = band_img.resize((ow, oh), pil_interp)
        out.append(np.array(band_resized) / 255.0)
    
    return np.stack(out, axis=2)


def _downsample_hsi(hsi: np.ndarray, sf: int) -> np.ndarray:
    h, w, _ = hsi.shape
    return _resize_hsi(hsi, (h // sf, w // sf), interpolation='area')


def _upsample_hsi(lr_hsi: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    return _resize_hsi(lr_hsi, out_hw, interpolation='bicubic')


def main() -> int:
    ap = argparse.ArgumentParser(description="HSRnet test (no cv2)")
    ap.add_argument("--dataset", default="CAVE", choices=["CAVE", "Harvard"])
    ap.add_argument("--sf", type=int, default=4, help="Scale factor")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--model_dir", default="methods/_HSRnet/models(cave)", help="Checkpoint directory")
    ap.add_argument("--crop", type=int, default=512, help="Center-crop size")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_images", type=int, default=12)
    ap.add_argument("--out_dir", type=str, default="")

    args = ap.parse_args()

    if tf.__version__.startswith("2"):
        tf.compat.v1.disable_eager_execution()

    gt_dir = Path(args.data_root) / "GT" / args.dataset
    ms_dir = Path(args.data_root) / "MS" / args.dataset

    gt_mats = sorted(glob.glob(str(gt_dir / "*.mat")))[:args.num_images]
    
    if not gt_mats:
        print(f"❌ No GT mats found in {gt_dir}")
        return 1

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # Build graph
    sz = args.crop if args.crop and args.crop > 0 else 512
    r_hp = tf.compat.v1.placeholder(shape=[1, sz, sz, 3], dtype=tf.float32)
    m_hp = tf.compat.v1.placeholder(shape=[1, sz // args.sf, sz // args.sf, 31], dtype=tf.float32)
    lms_p = tf.compat.v1.placeholder(shape=[1, sz, sz, 31], dtype=tf.float32)

    rs = rgbNet(m_hp, r_hp, reuse=False)
    out = tf.clip_by_value(tf.add(rs, lms_p), 0.0, 1.0)

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sums: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "sam": 0.0, "ergas": 0.0}

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(args.model_dir)
        if not ckpt:
            print(f"❌ No checkpoint found in {args.model_dir}")
            return 1
        
        saver.restore(sess, ckpt)
        print(f"✓ Loaded checkpoint: {ckpt}\n")

        for idx, gt_path in enumerate(gt_mats, start=1):
            name = Path(gt_path).stem
            
            try:
                gt_mat = scipy.io.loadmat(gt_path)
                if "hsi" in gt_mat:
                    gt_hsi = np.asarray(gt_mat["hsi"], dtype=np.float32)
                elif "ref" in gt_mat:
                    gt_hsi = np.asarray(gt_mat["ref"], dtype=np.float32)
                else:
                    print(f"  ⚠ {name}: No 'hsi' or 'ref' key")
                    continue

                ms_path = ms_dir / f"{name}.mat"
                if ms_path.exists():
                    msi = np.asarray(scipy.io.loadmat(str(ms_path))["msi"], dtype=np.float32)
                elif "msi" in gt_mat:
                    msi = np.asarray(gt_mat["msi"], dtype=np.float32)
                else:
                    msi = gt_hsi[:, :, :3]

                gt_hsi = normalize01(gt_hsi)
                msi = normalize01(msi)

                if args.crop and args.crop > 0:
                    gt_hsi = _center_crop(gt_hsi, (args.crop, args.crop))
                    msi = _center_crop(msi, (args.crop, args.crop))

                # Downsampling
                ms_down = _downsample_hsi(gt_hsi, args.sf)
                msi_up = _upsample_hsi(msi, (gt_hsi.shape[0], gt_hsi.shape[1]))

                # Prepare batches
                ms_batch = np.expand_dims(ms_down, 0)
                msi_batch = np.expand_dims(msi_up, 0)
                gt_batch = np.expand_dims(gt_hsi, 0)

                # Inference
                pred = sess.run(out, feed_dict={m_hp: ms_batch, r_hp: msi_batch[:, :, :, :3], lms_p: msi_batch})
                pred = np.clip(pred[0], 0, 1)

                # Metrics
                metrics = compute_metrics(gt_hsi, pred, sf=args.sf)
                psnr, ssim, sam, ergas = metrics['psnr'], metrics['ssim'], metrics['sam'], metrics['ergas']

                print(f"  {idx:2d}/{len(gt_mats)}: {name:30s} → PSNR={psnr:6.2f}  SSIM={ssim:.4f}  SAM={sam:6.2f}°")
                
                for key in sums:
                    sums[key] += metrics[key]

            except Exception as e:
                print(f"  ⚠ {name}: {e}")

        # Average
        n = len(gt_mats)
        print(f"\n{'='*80}")
        print(f"AVERAGE (SF={args.sf}, n={n}):")
        print(f"  PSNR: {sums['psnr']/n:.2f}")
        print(f"  SSIM: {sums['ssim']/n:.4f}")
        print(f"  SAM:  {sums['sam']/n:.2f}°")
        print(f"  ERGAS: {sums['ergas']/n:.2f}")
        print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    exit(main())
