#!/usr/bin/env python3
"""Test HSRnet on this repo's processed datasets (CAVE/Harvard).

HSRnet is TensorFlow 1.x code. This runner is designed to work under TF2
runtimes via tf.compat.v1.

It expects your repo dataset structure created by main/dataset_*.py:
  - data/GT/<dataset>/<name>.mat with GT HSI under key 'hsi' (CAVE) or 'ref' (Harvard)
  - data/MS/<dataset>/<name>.mat with MSI under key 'msi' (3 channels)

The model expects inputs:
  - ms: LR HSI (downsampled by sf)
  - lms: bicubic upsample of ms back to HR
  - rgb: HR MSI (3 channels)

If your Harvard images are not 512x512, use --crop 512 (default) to center-crop.

Example:
  python methods/_HSRnet/test_repo_dataset.py --dataset CAVE --sf 4 \
    --model_dir methods/_HSRnet/models(cave)

  python methods/_HSRnet/test_repo_dataset.py --dataset Harvard --sf 4 \
    --model_dir methods/_HSRnet/models(harvard)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import scipy.io
import tensorflow as tf

from methods._HSRnet.HSRnet import rgbNet
from tools.hif_metrics import compute_metrics, normalize01


def _center_crop(arr: np.ndarray, crop_hw: Tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[0], arr.shape[1]
    ch, cw = crop_hw
    if h < ch or w < cw:
        raise ValueError(f"Cannot crop {crop_hw} from {arr.shape}")
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return arr[y0 : y0 + ch, x0 : x0 + cw, ...]


def _resize_hsi(hsi: np.ndarray, out_hw: Tuple[int, int], *, interpolation) -> np.ndarray:
    oh, ow = out_hw
    out = []
    for b in range(hsi.shape[2]):
        out.append(cv2.resize(hsi[:, :, b], (ow, oh), interpolation=interpolation))
    return np.stack(out, axis=2)


def _downsample_hsi(hsi: np.ndarray, sf: int) -> np.ndarray:
    h, w, _ = hsi.shape
    return _resize_hsi(hsi, (h // sf, w // sf), interpolation=cv2.INTER_AREA)


def _upsample_hsi(lr_hsi: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    return _resize_hsi(lr_hsi, out_hw, interpolation=cv2.INTER_CUBIC)


def main() -> int:
    ap = argparse.ArgumentParser(description="HSRnet test runner for repo datasets")
    ap.add_argument("--dataset", required=True, choices=["CAVE", "Harvard"])
    ap.add_argument("--sf", type=int, default=4, help="Scale factor (HSRnet pretrained is typically sf=4)")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--model_dir", required=True, type=str, help="Checkpoint directory (e.g. methods/_HSRnet/models(cave))")
    ap.add_argument("--crop", type=int, default=512, help="Center-crop to this size (0 disables crop)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="", help="Optional directory to save outputs as .mat (key 'sri')")

    args = ap.parse_args()

    if tf.__version__.startswith("2"):
        tf.compat.v1.disable_eager_execution()

    gt_dir = Path(args.data_root) / "GT" / args.dataset
    ms_dir = Path(args.data_root) / "MS" / args.dataset

    gt_mats = sorted(glob.glob(str(gt_dir / "*.mat")))
    if not gt_mats:
        raise SystemExit(f"No GT mats found in {gt_dir}")

    if args.limit and args.limit > 0:
        gt_mats = gt_mats[: args.limit]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # Build graph for fixed crop size (recommended for HSRnet checkpoints)
    if args.crop and args.crop > 0:
        sz = args.crop
        r_hp = tf.compat.v1.placeholder(shape=[1, sz, sz, 3], dtype=tf.float32)
        m_hp = tf.compat.v1.placeholder(shape=[1, sz // args.sf, sz // args.sf, 31], dtype=tf.float32)
        lms_p = tf.compat.v1.placeholder(shape=[1, sz, sz, 31], dtype=tf.float32)
    else:
        # Still keep static channel dims
        r_hp = tf.compat.v1.placeholder(shape=[1, None, None, 3], dtype=tf.float32)
        m_hp = tf.compat.v1.placeholder(shape=[1, None, None, 31], dtype=tf.float32)
        lms_p = tf.compat.v1.placeholder(shape=[1, None, None, 31], dtype=tf.float32)

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
            raise SystemExit(f"No checkpoint found in {args.model_dir}")
        saver.restore(sess, ckpt)
        print("Loaded checkpoint:", ckpt)

        for idx, gt_path in enumerate(gt_mats, start=1):
            name = Path(gt_path).stem
            gt_mat = scipy.io.loadmat(gt_path)
            if "hsi" in gt_mat:
                gt_hsi = np.asarray(gt_mat["hsi"])
            elif "ref" in gt_mat:
                gt_hsi = np.asarray(gt_mat["ref"])
            else:
                raise SystemExit(f"Missing GT key in {gt_path}: {list(gt_mat.keys())}")

            ms_path = ms_dir / f"{name}.mat"
            if ms_path.exists():
                msi = np.asarray(scipy.io.loadmat(ms_path)["msi"])
            elif "msi" in gt_mat:
                msi = np.asarray(gt_mat["msi"])
            else:
                # Fallback: synthesize a 3ch proxy from HSI
                msi = gt_hsi[:, :, :3]

            gt_hsi = normalize01(gt_hsi)
            msi = normalize01(msi)

            if args.crop and args.crop > 0:
                gt_hsi = _center_crop(gt_hsi, (args.crop, args.crop))
                msi = _center_crop(msi, (args.crop, args.crop))

            lr = _downsample_hsi(gt_hsi, args.sf)
            lms = _upsample_hsi(lr, (gt_hsi.shape[0], gt_hsi.shape[1]))

            # Model expects batch dimension
            feed = {
                r_hp: msi[np.newaxis, ...].astype(np.float32),
                m_hp: lr[np.newaxis, ...].astype(np.float32),
                lms_p: lms[np.newaxis, ...].astype(np.float32),
            }
            pred = sess.run(out, feed_dict=feed)[0]

            m = compute_metrics(gt_hsi, pred, ratio=args.sf)
            for k in sums:
                sums[k] += m[k]

            if args.out_dir:
                scipy.io.savemat(str(Path(args.out_dir) / f"{name}.mat"), {"sri": pred})

            print(
                f"[{idx}/{len(gt_mats)}] {args.dataset}/{name} "
                f"PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} SAM={m['sam']:.4f} ERGAS={m['ergas']:.3f}"
            )

    n = float(len(gt_mats))
    avg = {k: v / n for k, v in sums.items()}
    print(
        f"\nAVERAGE ({args.dataset}, sf={args.sf}, n={len(gt_mats)}): "
        f"PSNR={avg['psnr']:.3f} SSIM={avg['ssim']:.4f} SAM={avg['sam']:.4f} ERGAS={avg['ergas']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
