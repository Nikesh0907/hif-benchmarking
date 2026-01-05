#!/usr/bin/env python3

import argparse
import os
from glob import glob

import numpy as np
import scipy.io as sio


def _bytes_feature(value):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 1.0:
        return np.clip(x, 0.0, 1.0)
    if mx <= 255.0:
        denom = 255.0
    elif mx <= 4095.0:
        denom = 4095.0
    elif mx <= 65535.0:
        denom = 65535.0
    else:
        denom = mx
    return np.clip(x / denom, 0.0, 1.0)


def _resize_hwc(img, out_h, out_w):
    try:
        import cv2
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception:
        from skimage.transform import resize as sk_resize

        return sk_resize(
            img,
            (out_h, out_w, img.shape[2]),
            order=1,
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Create DBIN training TFRecord from CAVE-style MATs')
    ap.add_argument('--gt_dir', required=True, help='Directory with GT mats containing key hsi (HxWx31).')
    ap.add_argument('--rgb_dir', required=False, default=None, help='Directory with RGB mats containing key rgb or msi (HxWx3). If missing, uses msi from gt mats or synthesizes RGB.')
    ap.add_argument('--out', required=True, help='Output TFRecord path')
    ap.add_argument('--sf', type=int, default=8, help='Scale factor (even >=2). Examples: 8,16,32.')
    ap.add_argument('--patch', type=int, default=64, help='Training patch size (must be divisible by sf).')
    ap.add_argument('--patches_per_image', type=int, default=200, help='Random patches sampled per image.')
    ap.add_argument('--seed', type=int, default=123, help='RNG seed')
    args = ap.parse_args()

    sf = int(args.sf)
    patch = int(args.patch)
    if sf < 2 or (sf % 2) != 0:
        raise SystemExit(f'Invalid sf={sf}; must be even and >=2')
    if patch % sf != 0:
        raise SystemExit(f'Invalid patch={patch}; must be divisible by sf={sf}')

    gt_paths = sorted(glob(os.path.join(args.gt_dir, '*.mat')))
    if not gt_paths:
        raise SystemExit(f'No .mat files found in {args.gt_dir}')

    rng = np.random.default_rng(args.seed)

    import tensorflow as tf
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    writer = tf.io.TFRecordWriter(args.out)

    total = 0
    for p in gt_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        m = sio.loadmat(p)
        if 'hsi' not in m:
            continue
        hsi = _normalize01(np.array(m['hsi'], dtype=np.float32))
        if hsi.ndim != 3 or hsi.shape[2] < 31:
            continue
        hsi = hsi[:, :, :31]

        rgb = None
        if args.rgb_dir:
            rp = os.path.join(args.rgb_dir, base + '.mat')
            if os.path.isfile(rp):
                rm = sio.loadmat(rp)
                if 'rgb' in rm:
                    rgb = _normalize01(np.array(rm['rgb'], dtype=np.float32))
                elif 'msi' in rm:
                    rgb = _normalize01(np.array(rm['msi'], dtype=np.float32))
        if rgb is None:
            if 'msi' in m:
                rgb = _normalize01(np.array(m['msi'], dtype=np.float32))
            elif 'rgb' in m:
                rgb = _normalize01(np.array(m['rgb'], dtype=np.float32))

        if rgb is None:
            idx_b, idx_g, idx_r = 7, 15, 23
            rgb = np.stack([hsi[..., idx_b], hsi[..., idx_g], hsi[..., idx_r]], axis=-1)

        H, W = hsi.shape[0], hsi.shape[1]
        if H < patch or W < patch:
            continue

        for _ in range(int(args.patches_per_image)):
            y = int(rng.integers(0, H - patch + 1))
            x = int(rng.integers(0, W - patch + 1))

            gt_patch = hsi[y:y + patch, x:x + patch, :]
            pan_patch = rgb[y:y + patch, x:x + patch, :]

            ms_size = patch // sf
            ms_patch = _resize_hwc(gt_patch, ms_size, ms_size)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'pan_raw': _bytes_feature(pan_patch.astype(np.float32).tobytes()),
                        'gt_raw': _bytes_feature(gt_patch.astype(np.float32).tobytes()),
                        'ms_raw': _bytes_feature(ms_patch.astype(np.float32).tobytes()),
                    }
                )
            )
            writer.write(example.SerializeToString())
            total += 1

    writer.close()
    print(f'Wrote {total} patches to {args.out}')


if __name__ == '__main__':
    main()
