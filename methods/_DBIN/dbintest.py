#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBIN/EDBIN evaluation (Python-only)

Runs the TensorFlow 1.x evaluation equivalent to test_boost_res.py without any
MATLAB dependency, using TFRecord test data and restoring a checkpoint.

Requirements:
- TensorFlow 1.15.x (tf.contrib is used in the original code)
- numpy, scipy, scikit-image, cv2 (OpenCV)

Usage example (Kaggle):
    python methods/_DBIN/dbintest.py /kaggle/input/<CAVE>/Data/Test/HSI \
        --model_dir methods/_DBIN/models_ibp_sn22 \
        --num_images 12
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio

import tensorflow as tf
try:
    import cv2  # optional; may fail on headless images missing libGL
except Exception:  # pragma: no cover
    cv2 = None
try:
    from skimage.measure import compare_ssim
except Exception:
    from skimage.metrics import structural_similarity as compare_ssim

# Ensure repo root is on sys.path so absolute imports work when run as a script
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

read_and_decode_test = None  # imported lazily inside main()
import scipy.io as sio


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _normalize01(x):
    """Best-effort normalization to [0,1] matching common HSI conventions.

    DBIN/EDBIN training/eval expects inputs in [0,1]. Many public CAVE/HSI mats
    are stored as uint16 or floats in 0..255/4095/65535.
    """
    x = np.asarray(x, dtype=np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 1.0:
        return np.clip(x, 0.0, 1.0)
    # Heuristics for common ranges
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


def _count_tfrecord_examples(path):
    # Fast-enough for small test sets; avoids silent hangs when num_images > records.
    try:
        it = tf.compat.v1.io.tf_record_iterator(path)
    except Exception:
        it = tf.python_io.tf_record_iterator(path)
    return sum(1 for _ in it)


def _find_candidate_key(mdict, want_channels):
    # Try common names first
    for cand in ['gt', 'hsi', 'HSI', 'cube', 'X', 'data']:
        if cand in mdict:
            arr = np.array(mdict[cand])
            if arr.ndim in (3, 4) and arr.shape[-1] >= want_channels:
                return cand
    # Fallback: largest 3D/4D array with channels >= want_channels
    best = None
    best_size = -1
    for k, v in mdict.items():
        arr = np.array(v)
        if arr.ndim in (3, 4) and arr.shape[-1] >= want_channels:
            size = arr.size
            if size > best_size:
                best = k
                best_size = size
    return best

def _load_mat(path, key, want_channels=None, allow_autodetect=False):
    m = sio.loadmat(path)
    if key in m:
        return np.array(m[key], dtype=np.float32)
    if allow_autodetect:
        cand = _find_candidate_key(m, want_channels or 1)
        if cand is not None:
            return np.array(m[cand], dtype=np.float32)
    raise KeyError('Key {} not found in {}. Available: {}'.format(key, path, list(m.keys())))


def convert_mats_to_test_tfrecord(mat_dir, save_path, rgb_dir=None, sf=8):
    """
        Create a test-style TFRecord with features expected by the test reader:
            pan_raw [H,W,3], pan2_raw [H/2,W/2,3], pan4_raw [H/4,W/4,3],
            gt_raw [H,W,31], ms_raw [H/sf,W/sf,31], ms2_raw [H/(sf/2),W/(sf/2),31].

    Supports two modes:
     1) Dataset-packed mats (gt.mat/ms.mat/pan.mat containing N samples)
     2) A directory of per-scene mats (e.g., jelly_beans.mat with key 'hsi', optional 'msi')

    Missing derived variants are generated via bilinear resize. If 'msi' is absent,
    a 3-channel proxy is synthesized from 'hsi' bands.
    """
    _ensure_dir(os.path.dirname(save_path))

    if sf < 2 or (sf % 2) != 0:
        raise ValueError('sf must be an even integer >= 2 (got {})'.format(sf))

    def ensure_4d(x):
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        return x.astype(np.float32)

    def resize_hw(arr, new_hw):
        # arr: [H,W,C] or [N,H,W,C]; returns same rank resized to new_hw
        if arr.ndim == 3:
            return cv2.resize(arr, new_hw, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        out = []
        for i in range(arr.shape[0]):
            out.append(cv2.resize(arr[i], new_hw, interpolation=cv2.INTER_LINEAR))
        return np.stack(out, axis=0).astype(np.float32)

    def downsample_hsi_to(arr_hwc, out_hw):
        # Better downsample for MS generation than bilinear: INTER_AREA is designed for shrink.
        h, w = out_hw
        out = []
        for c in range(arr_hwc.shape[2]):
            out.append(cv2.resize(arr_hwc[:, :, c], (w, h), interpolation=cv2.INTER_AREA))
        return np.stack(out, axis=2).astype(np.float32)

    def try_load_rgb_for_scene(scene_basename):
        if not rgb_dir:
            return None
        # 1) Try a MAT with key 'rgb' or 'msi'
        mat_path = os.path.join(rgb_dir, scene_basename + '.mat')
        if os.path.isfile(mat_path):
            try:
                m = sio.loadmat(mat_path)
                if 'rgb' in m:
                    return _normalize01(np.array(m['rgb'], dtype=np.float32))
                if 'msi' in m:
                    return _normalize01(np.array(m['msi'], dtype=np.float32))
            except Exception:
                pass
        # 2) Try common image extensions
        for ext in ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            img_path = os.path.join(rgb_dir, scene_basename + ext)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = np.repeat(img[:, :, None], 3, axis=2)
                # OpenCV loads BGR; convert to RGB for consistency with dataset scripts.
                if img.shape[2] >= 3:
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
                return _normalize01(img.astype(np.float32))
        return None

    gt_path = os.path.join(mat_dir, 'gt.mat')
    ms_path = os.path.join(mat_dir, 'ms.mat')
    pan_path = os.path.join(mat_dir, 'pan.mat')

    # Open TFRecord writer with TF2/TF1 compatibility
    try:
        writer = tf.io.TFRecordWriter(save_path)
    except Exception:
        writer = tf.python_io.TFRecordWriter(save_path)

    if os.path.isfile(gt_path):
        # Mode 1: dataset-packed mats
        gt = _load_mat(gt_path, 'gt', want_channels=31, allow_autodetect=True)
        ms = _load_mat(ms_path, 'ms', want_channels=31, allow_autodetect=True) if os.path.isfile(ms_path) else None
        pan = _load_mat(pan_path, 'pan', want_channels=3, allow_autodetect=True) if os.path.isfile(pan_path) else None

        gt = ensure_4d(_normalize01(gt))
        ms = ensure_4d(_normalize01(ms)) if ms is not None else None
        if pan is None:
            # synthesize RGB from gt bands
            idx_b, idx_g, idx_r = 7, 15, 23
            pan = np.stack([gt[..., idx_b], gt[..., idx_g], gt[..., idx_r]], axis=-1)
        pan = ensure_4d(_normalize01(pan))

        N = gt.shape[0]
        for i in range(N):
            H, W = gt.shape[1], gt.shape[2]
            if (H % sf) != 0 or (W % sf) != 0:
                raise ValueError('GT size {}x{} not divisible by sf={}'.format(H, W, sf))
            # derive downsamples
            pan2 = resize_hw(pan[i], (W // 2, H // 2))
            pan4 = resize_hw(pan[i], (W // 4, H // 4))
            gt_full = gt[i]
            gt2 = resize_hw(gt_full, (W // 2, H // 2))
            gt4 = resize_hw(gt_full, (W // 4, H // 4))
            ms_hw = (W // sf, H // sf)
            ms2_hw = (W // (sf // 2), H // (sf // 2))
            if ms is not None:
                ms_i = ms[i]
                if (ms_i.shape[0], ms_i.shape[1]) != (ms_hw[1], ms_hw[0]):
                    ms_i = resize_hw(ms_i, ms_hw)
            else:
                ms_i = resize_hw(gt_full, ms_hw)
            ms2 = resize_hw(ms_i, ms2_hw)

            # Safety: keep everything in [0,1]
            pan2 = _normalize01(pan2)
            pan4 = _normalize01(pan4)
            gt2 = _normalize01(gt2)
            gt4 = _normalize01(gt4)
            ms_i = _normalize01(ms_i)
            ms2 = _normalize01(ms2)

            example = tf.train.Example(features=tf.train.Features(feature={
                'pan_raw': _bytes_feature(pan[i].tobytes()),
                'pan2_raw': _bytes_feature(pan2.tobytes()),
                'pan4_raw': _bytes_feature(pan4.tobytes()),
                'gt_raw': _bytes_feature(gt_full.tobytes()),
                'ms_raw': _bytes_feature(ms_i.tobytes()),
                'ms2_raw': _bytes_feature(ms2.tobytes()),
            }))
            writer.write(example.SerializeToString())
    else:
        # Mode 2: per-scene mats in directory
        mats = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
        count = 0
        for fname in sorted(mats):
            path = os.path.join(mat_dir, fname)
            m = sio.loadmat(path)
            # GT candidate (31+ channels)
            gt_key = _find_candidate_key(m, want_channels=31)
            if gt_key is None:
                # skip files without a valid HSI
                continue
            gt = _normalize01(np.array(m[gt_key], dtype=np.float32))
            scene_base = os.path.splitext(fname)[0]
            if gt.ndim == 4:
                # handle packed samples inside one file
                for i in range(gt.shape[0]):
                    gt_i = _normalize01(gt[i])
                    H, W = gt_i.shape[0], gt_i.shape[1]
                    # PAN candidate (3+ channels)
                    pan_key = None
                    for pk in ['msi', 'pan']:
                        if pk in m:
                            pan_key = pk
                            break
                    pan_i = _normalize01(np.array(m[pan_key][i], dtype=np.float32)) if pan_key else None
                    if pan_i is None:
                        pan_i = try_load_rgb_for_scene(scene_base)
                    if pan_i is None:
                        # Last resort: synthesize RGB from 3 HSI bands (often worse than true MSI)
                        idx_b, idx_g, idx_r = 7, 15, 23
                        pan_i = _normalize01(np.stack([gt_i[..., idx_b], gt_i[..., idx_g], gt_i[..., idx_r]], axis=-1))

                    pan2 = resize_hw(pan_i, (W // 2, H // 2))
                    pan4 = resize_hw(pan_i, (W // 4, H // 4))
                    gt2 = resize_hw(gt_i, (W // 2, H // 2))
                    gt4 = resize_hw(gt_i, (W // 4, H // 4))
                    if (H % sf) != 0 or (W % sf) != 0:
                        raise ValueError('GT size {}x{} not divisible by sf={}'.format(H, W, sf))
                    ms_i = downsample_hsi_to(gt_i, (H // sf, W // sf))
                    ms2 = resize_hw(ms_i, (W // (sf // 2), H // (sf // 2)))

                    pan2 = _normalize01(pan2)
                    pan4 = _normalize01(pan4)
                    ms_i = _normalize01(ms_i)
                    ms2 = _normalize01(ms2)

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'pan_raw': _bytes_feature(pan_i.tobytes()),
                        'pan2_raw': _bytes_feature(pan2.tobytes()),
                        'pan4_raw': _bytes_feature(pan4.tobytes()),
                        'gt_raw': _bytes_feature(gt_i.tobytes()),
                        'ms_raw': _bytes_feature(ms_i.tobytes()),
                        'ms2_raw': _bytes_feature(ms2.tobytes()),
                    }))
                    writer.write(example.SerializeToString())
                    count += 1
            else:
                # single sample per file
                H, W = gt.shape[0], gt.shape[1]
                pan_key = None
                for pk in ['msi', 'pan']:
                    if pk in m:
                        pan_key = pk
                        break
                pan = _normalize01(np.array(m[pan_key], dtype=np.float32)) if pan_key else None
                if pan is None:
                    pan = try_load_rgb_for_scene(scene_base)
                if pan is None:
                    idx_b, idx_g, idx_r = 7, 15, 23
                    pan = _normalize01(np.stack([gt[..., idx_b], gt[..., idx_g], gt[..., idx_r]], axis=-1))

                pan2 = resize_hw(pan, (W // 2, H // 2))
                pan4 = resize_hw(pan, (W // 4, H // 4))
                gt2 = resize_hw(gt, (W // 2, H // 2))
                gt4 = resize_hw(gt, (W // 4, H // 4))
                if (H % sf) != 0 or (W % sf) != 0:
                    raise ValueError('GT size {}x{} not divisible by sf={}'.format(H, W, sf))
                ms = downsample_hsi_to(gt, (H // sf, W // sf))
                ms2 = resize_hw(ms, (W // (sf // 2), H // (sf // 2)))

                pan2 = _normalize01(pan2)
                pan4 = _normalize01(pan4)
                ms = _normalize01(ms)
                ms2 = _normalize01(ms2)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'pan_raw': _bytes_feature(pan.tobytes()),
                    'pan2_raw': _bytes_feature(pan2.tobytes()),
                    'pan4_raw': _bytes_feature(pan4.tobytes()),
                    'gt_raw': _bytes_feature(gt.tobytes()),
                    'ms_raw': _bytes_feature(ms.tobytes()),
                    'ms2_raw': _bytes_feature(ms2.tobytes()),
                }))
                writer.write(example.SerializeToString())
                count += 1

        print('Wrote {} samples to {}'.format(count, save_path))

    writer.close()
    return save_path


def _resolve_checkpoint(model_dir):
    """Return a checkpoint prefix path from a directory, even if 'checkpoint' file is missing/stale.

    Tries tf.compat.v1.train.latest_checkpoint first. If that fails, scans for
    '*.ckpt.index' files and picks the numerically largest step that has a
    matching data shard next to it.
    """
    import re

    # 1) Try TensorFlow's standard mechanism
    ckpt = tf.compat.v1.train.latest_checkpoint(model_dir)
    if ckpt:
        idx = ckpt + ".index"
        dat = ckpt + ".data-00000-of-00001"
        if os.path.exists(idx) and os.path.exists(dat):
            return ckpt

    # 2) Fallback: scan directory for checkpoint shards
    candidates = []
    for fname in os.listdir(model_dir):
        if fname.endswith('.ckpt.index'):
            prefix = fname[:-len('.index')]  # remove only the trailing '.index'
            # Ensure data shard exists
            dat = os.path.join(model_dir, prefix + '.data-00000-of-00001')
            if not os.path.exists(dat):
                continue
            m = re.search(r'model-(\d+)\.ckpt$', prefix)
            step = int(m.group(1)) if m else -1
            candidates.append((step, os.path.join(model_dir, prefix)))

    if candidates:
        candidates.sort()
        return candidates[-1][1]

    raise RuntimeError(
        'No valid checkpoint found in {}. Ensure .index and .data-00000-of-00001 files are present.'.format(model_dir)
    )


def compute_ms_ssim(image1, image2):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]
    n = image1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        # For float images, specify data_range; outputs are clipped to [0,1]
        single_ssim = compare_ssim(image1[:, :, i], image2[:, :, i], data_range=1.0)
        ms_ssim += single_ssim
    return ms_ssim / n


def compute_sam(image1, image2):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    if image1.ndim == 4:
        image1 = image1[0]
    if image2.ndim == 4:
        image2 = image2[0]
    h, w, c = image1.shape
    image1 = np.reshape(image1, (h * w, c))
    image2 = np.reshape(image2, (h * w, c))
    mole = np.sum(np.multiply(image1, image2), axis=1)
    image1_norm = np.sqrt(np.sum(np.square(image1), axis=1))
    image2_norm = np.sqrt(np.sum(np.square(image2), axis=1))
    deno = np.multiply(image1_norm, image2_norm)
    sam = np.rad2deg(np.arccos((mole + 1e-11) / (deno + 1e-11)))
    return np.mean(sam)


def compute_ergas(mse, out, sf=8):
    out = np.asarray(out)
    if out.ndim == 4:
        out = out[0]
    h, w, c = out.shape
    out = np.reshape(out, (h * w, c))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (c, 1))
    out_mean = np.reshape(out_mean, (c, 1))
    ergas = 100.0 / float(sf) * np.sqrt(np.mean(mse / (out_mean ** 2 + 1e-12)))
    return ergas


def read_and_decode_test_sf(tfrecords_file, batch_size, image_size, sf):
    """Read test TFRecord with parameterized shapes.

    Keeps the original queue-based pipeline (TF1 style) for Kaggle/TF2 compat,
    but avoids hard-coded 512/64/128 so we can vary `sf`.
    """
    if sf < 2 or (sf % 2) != 0:
        raise ValueError('sf must be an even integer >= 2 (got {})'.format(sf))
    ms_size = image_size // sf
    ms2_size = image_size // (sf // 2)

    filename_queue = tf.compat.v1.train.string_input_producer([tfrecords_file])
    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.compat.v1.parse_single_example(
        serialized_example,
        features={
            'pan_raw': tf.io.FixedLenFeature([], tf.string),
            'pan2_raw': tf.io.FixedLenFeature([], tf.string),
            'pan4_raw': tf.io.FixedLenFeature([], tf.string),
            'gt_raw': tf.io.FixedLenFeature([], tf.string),
            'ms_raw': tf.io.FixedLenFeature([], tf.string),
            'ms2_raw': tf.io.FixedLenFeature([], tf.string),
        },
    )

    pan = tf.io.decode_raw(img_features['pan_raw'], tf.float32)
    pan = tf.reshape(pan, [image_size, image_size, 3])
    pan2 = tf.io.decode_raw(img_features['pan2_raw'], tf.float32)
    pan2 = tf.reshape(pan2, [image_size // 2, image_size // 2, 3])
    pan4 = tf.io.decode_raw(img_features['pan4_raw'], tf.float32)
    pan4 = tf.reshape(pan4, [image_size // 4, image_size // 4, 3])
    gt = tf.io.decode_raw(img_features['gt_raw'], tf.float32)
    gt = tf.reshape(gt, [image_size, image_size, 31])
    ms = tf.io.decode_raw(img_features['ms_raw'], tf.float32)
    ms = tf.reshape(ms, [ms_size, ms_size, 31])
    ms2 = tf.io.decode_raw(img_features['ms2_raw'], tf.float32)
    ms2 = tf.reshape(ms2, [ms2_size, ms2_size, 31])

    pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch = tf.compat.v1.train.batch(
        [pan, pan2, pan4, gt, ms, ms2],
        batch_size=batch_size,
        num_threads=4,
        capacity=300,
        allow_smaller_final_batch=False,
    )
    return pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch


def _vsi():
    # Prefer TF1 initializer to avoid Keras 3 dtype/ref issues
    try:
        return tf.compat.v1.variance_scaling_initializer()
    except Exception:
        return tf.compat.v1.glorot_uniform_initializer()


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[1, 2])


def dense_manual(x, units, activation_fn=None, scope='dense'):
    """Keras-3-safe dense layer with TF1-style variable names.

    Matches the variable naming used by `tf.layers.dense(name=...)` in TF1:
    `<scope>/kernel` and `<scope>/bias`.
    """
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        in_dim = x.get_shape().as_list()[-1]
        if in_dim is None:
            raise ValueError('dense_manual requires a static last dimension')
        w = tf.compat.v1.get_variable(
            'kernel',
            shape=[in_dim, units],
            initializer=_vsi(),
        )
        b = tf.compat.v1.get_variable(
            'bias',
            shape=[units],
            initializer=tf.zeros_initializer(),
        )
        y = tf.matmul(x, w) + b
        if activation_fn is not None:
            y = activation_fn(y)
        return y


def spectral_norm(w, iteration=1):
    # Matches methods/_DBIN/utils3.py naming and behavior
    w_shape = w.shape.as_list()
    w_reshaped = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable(
        'u',
        [1, w_shape[-1]],
        initializer=tf.random_normal_initializer(),
        trainable=False,
    )

    u_hat = u
    v_hat = None
    for _ in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w_reshaped)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w_reshaped / sigma * 0.7
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def conv_sn(x, channels, weight_decay, kernel=3, stride=1, use_bias=True, scope='conv'):
    # Drop-in for methods/_DBIN/utils3.py::conv (kernel/bias/u variable names)
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        w = tf.compat.v1.get_variable(
            'kernel',
            shape=[kernel, kernel, x.get_shape()[-1], channels],
            initializer=_vsi(),
        )
        bias = tf.compat.v1.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0))
        y = tf.nn.conv2d(input=x, filters=spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME')
        if use_bias:
            y = tf.nn.bias_add(y, bias)
        return y


def upsample(x, ref):
    """Bilinear upsample to match spatial size of `ref`.

    The provided `models_ibp_sn22` checkpoint does not contain any `up_net`/CARAFE
    variables, so upsampling must be parameter-free for full restoration.
    """
    target_hw = tf.shape(ref)[1:3]
    return tf.image.resize(x, target_hw, method=tf.image.ResizeMethod.BILINEAR)


    def _resize_hw(image, new_h, new_w):
        """Resize HxWxC float32 image.

        Prefers OpenCV INTER_AREA for downsampling, but falls back to skimage when
        OpenCV is unavailable (e.g., missing libGL in headless containers).
        """
        if cv2 is not None:
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        from skimage.transform import resize as sk_resize

        return sk_resize(
            image,
            (new_h, new_w, image.shape[2]),
            order=1,
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)


def Fusion(Z, Y, weight_decay, num_spectral=31, num_fm=64, reuse=True):
    # Exact structure from methods/_DBIN/train_cave_edbin.py
    with tf.compat.v1.variable_scope('py'):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()

        lms = upsample(Y, Z)
        Xin = tf.concat([lms, Z], axis=3)

        Xt = conv_sn(Xin, num_fm, weight_decay, scope='in')
        Xt = lrelu(Xt)

        for i in range(4):
            Xi = conv_sn(Xt, num_fm, weight_decay, scope='res{}1'.format(i))
            Xi = lrelu(Xi)
            Xi = conv_sn(Xi, num_fm, weight_decay, scope='res{}2'.format(i))

            mask = global_avg_pool(Xi)
            mask = dense_manual(mask, units=num_fm // 16, activation_fn=tf.nn.relu, scope='se{}1'.format(i))
            mask = dense_manual(mask, units=num_fm, activation_fn=None, scope='se{}2'.format(i))
            mask = tf.reshape(mask, [-1, 1, 1, num_fm])
            mask = tf.sigmoid(mask)
            Xi = tf.multiply(Xi, mask)

            Xt = Xt + Xi

        X = conv_sn(Xt, num_spectral, weight_decay, scope='out')
        return X


def boost_lap(X, Z_in, Y_in, weight_decay, num_spectral=31, num_fm=64, sf=8, reuse=True):
    with tf.compat.v1.variable_scope('recursive'):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()

        Z = conv_sn(X, 3, weight_decay, scope='dz')
        Z = lrelu(Z)

        # Downsample by the scale factor. Original DBIN uses kernel=12, stride=8 for sf=8.
        # Keep a similar receptive field by defaulting to kernel = sf + 4.
        dy_kernel = int(sf) + 4
        Y = conv_sn(X, num_spectral, weight_decay, kernel=dy_kernel, stride=int(sf), scope='dy')
        Y = lrelu(Y)

        dZ = Z_in - Z
        dY = Y_in - Y

        dX = Fusion(dZ, dY, weight_decay, num_spectral=num_spectral, num_fm=num_fm, reuse=True)
        X = X + dX
        return X


def fusion_net(Z, Y, num_spectral=31, num_fm=64, num_ite=8, sf=8, reuse=False, weight_decay=1e-5):
    # Exact structure from methods/_DBIN/train_cave_edbin.py
    with tf.compat.v1.variable_scope('fusion_net'):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()

        X = Fusion(Z, Y, weight_decay, num_spectral=num_spectral, num_fm=num_fm, reuse=False)
        Xs = X

        for _ in range(num_ite):
            X = boost_lap(
                X,
                Z,
                Y,
                weight_decay,
                num_spectral=num_spectral,
                num_fm=num_fm,
                sf=sf,
                reuse=True,
            )
            Xs = tf.concat([Xs, X], axis=3)

        # train_cave_edbin.py uses utils3.conv with scope='out_conv' and use_bias=False.
        # utils3.conv still creates a 'bias' variable, it just doesn't add it.
        X = conv_sn(Xs, num_spectral, weight_decay, use_bias=False, scope='out_conv')
        return X


def main():
    parser = argparse.ArgumentParser(description='Python-only DBIN evaluation')
    # Keep positional arg for backwards compatibility
    parser.add_argument('data', nargs='?', default=None, type=str,
                        help='Path to test TFRecord file OR a directory containing mat files')
    # Also accept explicit flags (friendlier in notebooks)
    parser.add_argument('--tfrecord', dest='tfrecord', type=str, default=None,
                        help='Explicit TFRecord path (overrides positional data)')
    parser.add_argument('--mat_dir', dest='mat_dir', type=str, default=None,
                        help='Explicit MAT directory path (overrides positional data)')
    parser.add_argument('--rgb_dir', dest='rgb_dir', type=str, default=None,
                        help='Optional directory with matching RGB/MSI per scene (same basename). Helps match DBIN training data and improves metrics.')
    parser.add_argument('--model_dir', type=str, default=os.path.join('methods', '_DBIN', 'models_ibp_sn22'), help='Checkpoint directory (default: methods/_DBIN/models_ibp_sn22)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_images', type=int, default=12, help='Number of test samples to evaluate. Use 0 to auto-count.')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--sf', type=int, default=8, help='Scale factor between GT and MS (default: 8 for CAVE). Must be even.')
    args = parser.parse_args()

    global read_and_decode_test
    if read_and_decode_test is None:
        # Import the TFRecord reader for test data with a robust fallback
        try:
            from methods._DBIN.mat_convert_to_tfrecord_p_end import read_and_decode_test as _reader
        except ModuleNotFoundError:
            import importlib.util
            reader_path = os.path.join(FILE_DIR, 'mat_convert_to_tfrecord_p_end.py')
            spec = importlib.util.spec_from_file_location('mat_reader', reader_path)
            mat_reader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mat_reader)
            _reader = mat_reader.read_and_decode_test
        read_and_decode_test = _reader

    # Enable TF1 behavior under TF2 runtimes
    if tf.__version__.startswith('2'):
        tf.compat.v1.disable_eager_execution()

    # Kaggle often runs CPU-only; avoid CUDA init noise unless user explicitly wants GPU.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    batch_size = args.batch_size
    image_size = args.image_size
    weight_decay = args.weight_decay
    sf = args.sf
    if sf < 2 or (sf % 2) != 0:
        raise SystemExit('Invalid --sf {} (must be even and >=2)'.format(sf))

    # Placeholders
    gt_holder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 31])
    ms_holder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, image_size // sf, image_size // sf, 31])
    pan_holder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
    pan2_holder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, image_size // 2, image_size // 2, 3])
    pan4_holder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, image_size // 4, image_size // 4, 3])

    # Resolve input path precedence: --tfrecord / --mat_dir / positional
    data_path = args.tfrecord or args.mat_dir or args.data
    if not data_path:
        raise SystemExit('Missing input: provide positional data path, or --tfrecord, or --mat_dir')

    # If input is a directory, auto-convert mats to a test TFRecord first
    wrote_count = None
    if os.path.isdir(data_path):
        out_dir = os.path.join('methods', '_DBIN', 'training_data')
        _ensure_dir(out_dir)
        tfrecord_path = os.path.join(out_dir, 'autotest.tfrecords')
        print('Converting MATs in {} -> {}'.format(data_path, tfrecord_path))
        convert_mats_to_test_tfrecord(data_path, tfrecord_path, rgb_dir=args.rgb_dir, sf=sf)
        data_path = tfrecord_path

        # If user requested auto-count, we can infer after conversion.
        if args.num_images == 0:
            wrote_count = _count_tfrecord_examples(data_path)
            args.num_images = wrote_count
            print('Auto-counted {} TFRecord samples'.format(args.num_images))
    elif args.num_images == 0:
        args.num_images = _count_tfrecord_examples(data_path)
        print('Auto-counted {} TFRecord samples'.format(args.num_images))

    # Dataset pipeline
    pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch = read_and_decode_test_sf(
        data_path, batch_size=batch_size, image_size=image_size, sf=sf
    )

    # Build model
    X = fusion_net(
        pan_holder,
        ms_holder,
        num_spectral=31,
        num_fm=64,
        num_ite=8,
        sf=sf,
        reuse=False,
        weight_decay=weight_decay,
    )
    output = tf.clip_by_value(X, 0.0, 1.0)

    mse = tf.square(output - gt_holder)
    mse = tf.reshape(mse, [image_size, image_size, 31])
    final_mse = tf.reduce_mean(mse, axis=[0, 1])

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    init = tf.group(
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer(),
    )
    saver = tf.compat.v1.train.Saver()

    average_psnr = 0.0
    average_ssim = 0.0
    average_sam = 0.0
    average_ergas = 0.0

    gt_out = np.zeros(shape=[args.num_images, image_size, image_size, 31], dtype=np.float32)
    net_out = np.zeros(shape=[args.num_images, image_size, image_size, 31], dtype=np.float32)

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)

        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # Restore checkpoint using robust resolution of checkpoint prefix
        try:
            ckpt = _resolve_checkpoint(args.model_dir)
        except Exception as e:
            raise RuntimeError('No checkpoint could be resolved in {}: {}'.format(args.model_dir, e))

        try:
            saver.restore(sess, ckpt)
            print('Loaded checkpoint:', ckpt)
        except Exception as e:
            print('Standard restore failed; attempting exact-name restore:', str(e))
            reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
            ckpt_vars = set(reader.get_variable_to_shape_map().keys())
            graph_vars = tf.compat.v1.trainable_variables()
            exact_map = {}
            skipped = []
            for v in graph_vars:
                name = v.name.split(':')[0]
                if name in ckpt_vars:
                    exact_map[name] = v
                else:
                    skipped.append(name)
            print('Will restore {} vars from checkpoint; skipping {} (not present).'.format(len(exact_map), len(skipped)))
            if skipped:
                print('Skipped examples:', skipped[:10])
            if exact_map:
                saver_exact = tf.compat.v1.train.Saver(var_list=exact_map)
                saver_exact.restore(sess, ckpt)
                print('Loaded checkpoint with exact-name restore:', ckpt)
            else:
                raise RuntimeError('No overlapping variables between graph and checkpoint {}.'.format(ckpt))

        for i in range(args.num_images):
            pan, pan2, pan4, gt, ms, ms2 = sess.run([pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch])
            out, mse_loss = sess.run(
                [output, final_mse],
                feed_dict={
                    pan_holder: pan,
                    pan2_holder: pan2,
                    pan4_holder: pan4,
                    gt_holder: gt,
                    ms_holder: ms,
                },
            )

            gt_out[i, :, :, :] = gt
            net_out[i, :, :, :] = np.array(out)

            mse_loss = np.array(mse_loss)
            ms_psnr = np.mean(10 * np.log10(1.0 / mse_loss))
            temp_ssim = compute_ms_ssim(out, gt)
            temp_sam = compute_sam(out, gt)
            temp_ergas = compute_ergas(mse_loss, out, sf=sf)

            print(
                'image{} temp_psnr: {:.4f}, temp_ssim: {:.4f}, temp_sam: {:.4f}, temp_ergas: {:.4f}'.format(
                    i, ms_psnr, temp_ssim, temp_sam, temp_ergas
                )
            )

            average_psnr += ms_psnr / float(args.num_images)
            average_ssim += temp_ssim / float(args.num_images)
            average_sam += temp_sam / float(args.num_images)
            average_ergas += temp_ergas / float(args.num_images)

        coord.request_stop()
        coord.join(threads)

    os.makedirs('./result', exist_ok=True)
    sio.savemat('./result/out.mat', {'gt_out': gt_out, 'net_out': net_out})
    print(
        'average_psnr: {:.4f}, average_ssim: {:.4f}, average_sam: {:.4f}, average_ergas: {:.4f}'.format(
            average_psnr, average_ssim, average_sam, average_ergas
        )
    )


if __name__ == '__main__':
    main()
