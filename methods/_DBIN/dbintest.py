#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBIN/EDBIN evaluation (Python-only)

Runs the TensorFlow 1.x evaluation equivalent to test_boost_res.py without any
MATLAB dependency, using TFRecord test data and restoring a checkpoint.

Requirements:
- TensorFlow 1.15.x (tf.contrib is used in the original code)
- numpy, scipy, scikit-image, cv2 (OpenCV)

Usage example:
    python methods/_DBIN/dbintest.py \
        --tfrecord methods/_DBIN/training_data/testp.tfrecords \
        --model_dir methods/_DBIN/models_ibp_sn22 \
        --num_images 12
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio

import tensorflow as tf
import tensorflow.contrib.layers as ly
import cv2
try:
    from skimage.measure import compare_ssim
except Exception:
    from skimage.metrics import structural_similarity as compare_ssim

# Import the repo's TFRecord reader for test data (absolute path import for script execution)
from methods._DBIN.mat_convert_to_tfrecord_p_end import read_and_decode_test
import scipy.io as sio


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


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
    def convert_mats_to_test_tfrecord(mat_dir, save_path):
        """
        Create a test-style TFRecord with features expected by read_and_decode_test:
          pan_raw [512,512,3], pan2_raw [256,256,3], pan4_raw [128,128,3],
          gt_raw [512,512,31], ms_raw [64,64,31], ms2_raw [128,128,31].

        Supports two modes:
         1) Dataset-packed mats (gt.mat/ms.mat/pan.mat containing N samples)
         2) A directory of per-scene mats (e.g., jelly_beans.mat with key 'hsi', optional 'msi')

        Missing derived variants are generated via bilinear resize. If 'msi' is absent,
        a 3-channel proxy is synthesized from 'hsi' bands.
        """
        _ensure_dir(os.path.dirname(save_path))

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

        gt_path = os.path.join(mat_dir, 'gt.mat')
        ms_path = os.path.join(mat_dir, 'ms.mat')
        pan_path = os.path.join(mat_dir, 'pan.mat')

        writer = tf.python_io.TFRecordWriter(save_path)

        if os.path.isfile(gt_path):
            # Mode 1: dataset-packed mats
            gt = _load_mat(gt_path, 'gt', want_channels=31, allow_autodetect=True)
            ms = _load_mat(ms_path, 'ms', want_channels=31, allow_autodetect=True) if os.path.isfile(ms_path) else None
            pan = _load_mat(pan_path, 'pan', want_channels=3, allow_autodetect=True) if os.path.isfile(pan_path) else None

            gt = ensure_4d(gt)
            ms = ensure_4d(ms) if ms is not None else None
            if pan is None:
                # synthesize RGB from gt bands
                idx_b, idx_g, idx_r = 7, 15, 23
                pan = np.stack([gt[..., idx_b], gt[..., idx_g], gt[..., idx_r]], axis=-1)
            pan = ensure_4d(pan)

            N = gt.shape[0]
            for i in range(N):
                H, W = gt.shape[1], gt.shape[2]
                # derive downsamples
                pan2 = resize_hw(pan[i], (W // 2, H // 2))
                pan4 = resize_hw(pan[i], (W // 4, H // 4))
                gt_full = gt[i]
                gt2 = resize_hw(gt_full, (W // 2, H // 2))
                gt4 = resize_hw(gt_full, (W // 4, H // 4))
                ms_i = ms[i] if ms is not None else resize_hw(gt_full, (64, 64))
                ms2 = resize_hw(ms_i, (128, 128))

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
                gt = np.array(m[gt_key], dtype=np.float32)
                if gt.ndim == 4:
                    # handle packed samples inside one file
                    for i in range(gt.shape[0]):
                        gt_i = gt[i]
                        H, W = gt_i.shape[0], gt_i.shape[1]
                        # PAN candidate (3+ channels)
                        pan_key = None
                        for pk in ['msi', 'pan']:
                            if pk in m:
                                pan_key = pk
                                break
                        pan_i = np.array(m[pan_key][i], dtype=np.float32) if pan_key else None
                        if pan_i is None:
                            idx_b, idx_g, idx_r = 7, 15, 23
                            pan_i = np.stack([gt_i[..., idx_b], gt_i[..., idx_g], gt_i[..., idx_r]], axis=-1)

                        pan2 = resize_hw(pan_i, (W // 2, H // 2))
                        pan4 = resize_hw(pan_i, (W // 4, H // 4))
                        gt2 = resize_hw(gt_i, (W // 2, H // 2))
                        gt4 = resize_hw(gt_i, (W // 4, H // 4))
                        ms_i = resize_hw(gt_i, (64, 64))
                        ms2 = resize_hw(ms_i, (128, 128))

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
                    pan = np.array(m[pan_key], dtype=np.float32) if pan_key else None
                    if pan is None:
                        idx_b, idx_g, idx_r = 7, 15, 23
                        pan = np.stack([gt[..., idx_b], gt[..., idx_g], gt[..., idx_r]], axis=-1)

                    pan2 = resize_hw(pan, (W // 2, H // 2))
                    pan4 = resize_hw(pan, (W // 4, H // 4))
                    gt2 = resize_hw(gt, (W // 2, H // 2))
                    gt4 = resize_hw(gt, (W // 4, H // 4))
                    ms = resize_hw(gt, (64, 64))
                    ms2 = resize_hw(ms, (128, 128))

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
    _ensure_dir(os.path.dirname(save_path))
    writer = tf.python_io.TFRecordWriter(save_path)
    for i in range(N):
        example = tf.train.Example(features=tf.train.Features(feature={
            'pan_raw': _bytes_feature(pan[i].tobytes()),
            'pan2_raw': _bytes_feature(pan2[i].tobytes()),
            'pan4_raw': _bytes_feature(pan4[i].tobytes()),
            'gt_raw': _bytes_feature(gt[i].tobytes()),
            'ms_raw': _bytes_feature(ms[i].tobytes()),
            'ms2_raw': _bytes_feature(ms2[i].tobytes()),
        }))
        writer.write(example.SerializeToString())
    writer.close()
    return save_path


def compute_ms_ssim(image1, image2):
    image1 = np.reshape(image1, (512, 512, 31))
    image2 = np.reshape(image2, (512, 512, 31))
    n = image1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = compare_ssim(image1[:, :, i], image2[:, :, i])
        ms_ssim += single_ssim
    return ms_ssim / n


def compute_sam(image1, image2):
    image1 = np.reshape(image1, (512 * 512, 31))
    image2 = np.reshape(image2, (512 * 512, 31))
    mole = np.sum(np.multiply(image1, image2), axis=1)
    image1_norm = np.sqrt(np.sum(np.square(image1), axis=1))
    image2_norm = np.sqrt(np.sum(np.square(image2), axis=1))
    deno = np.multiply(image1_norm, image2_norm)
    sam = np.rad2deg(np.arccos((mole + 1e-11) / (deno + 1e-11)))
    return np.mean(sam)


def compute_ergas(mse, out):
    out = np.reshape(out, (512 * 512, 31))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (31, 1))
    out_mean = np.reshape(out_mean, (31, 1))
    ergas = 100 / 8 * np.sqrt(np.mean(mse / out_mean ** 2))
    return ergas


def Fusion(Z, Y, num_spectral=31, num_fm=128, reuse=True, weight_decay=2e-5):
    with tf.variable_scope('py'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        lms = ly.conv2d_transpose(
            Y,
            num_outputs=num_spectral,
            kernel_size=12,
            stride=8,
            activation_fn=None,
            weights_initializer=ly.variance_scaling_initializer(),
            weights_regularizer=ly.l2_regularizer(weight_decay),
            reuse=tf.AUTO_REUSE,
            scope="lms",
        )

        Xin = tf.concat([lms, Z], axis=3)
        Xt = ly.conv2d(
            Xin,
            num_outputs=num_fm,
            kernel_size=3,
            stride=1,
            weights_regularizer=ly.l2_regularizer(weight_decay),
            weights_initializer=ly.variance_scaling_initializer(),
            activation_fn=tf.nn.leaky_relu,
            reuse=tf.AUTO_REUSE,
            scope="in",
        )

        for i in range(4):
            Xi = ly.conv2d(
                Xt,
                num_outputs=num_fm,
                kernel_size=3,
                stride=1,
                weights_regularizer=ly.l2_regularizer(weight_decay),
                weights_initializer=ly.variance_scaling_initializer(),
                activation_fn=tf.nn.leaky_relu,
                reuse=tf.AUTO_REUSE,
                scope="res" + str(i) + "1",
            )
            Xi = ly.conv2d(
                Xi,
                num_outputs=num_fm,
                kernel_size=3,
                stride=1,
                weights_regularizer=ly.l2_regularizer(weight_decay),
                weights_initializer=ly.variance_scaling_initializer(),
                activation_fn=tf.nn.leaky_relu,
                reuse=tf.AUTO_REUSE,
                scope="res" + str(i) + "2",
            )
            Xt = Xt + Xi

        X = ly.conv2d(
            Xt,
            num_outputs=num_spectral,
            kernel_size=3,
            stride=1,
            weights_regularizer=ly.l2_regularizer(weight_decay),
            weights_initializer=ly.variance_scaling_initializer(),
            activation_fn=tf.nn.leaky_relu,
            reuse=tf.AUTO_REUSE,
            scope="out",
        )

        return X


def boost_lap(X, Z_in, Y_in, num_spectral=31, num_fm=128, reuse=True, weight_decay=2e-5):
    with tf.variable_scope('recursive'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        Z = ly.conv2d(
            X,
            num_outputs=3,
            kernel_size=3,
            stride=1,
            weights_regularizer=ly.l2_regularizer(weight_decay),
            weights_initializer=ly.variance_scaling_initializer(),
            activation_fn=tf.nn.leaky_relu,
            reuse=tf.AUTO_REUSE,
            scope='dz',
        )

        Y = ly.conv2d(
            X,
            num_outputs=num_spectral,
            kernel_size=12,
            stride=8,
            weights_regularizer=ly.l2_regularizer(weight_decay),
            weights_initializer=ly.variance_scaling_initializer(),
            activation_fn=tf.nn.leaky_relu,
            reuse=tf.AUTO_REUSE,
            scope='dy',
        )

        dZ = Z_in - Z
        dY = Y_in - Y

        dX = Fusion(dZ, dY, num_spectral=num_spectral, num_fm=num_fm, reuse=True, weight_decay=weight_decay)
        X = X + dX
        return X


def fusion_net(Z, Y, num_spectral=31, num_fm=128, num_ite=5, reuse=False, weight_decay=2e-5):
    with tf.variable_scope('fusion_net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X = Fusion(Z, Y, num_spectral=num_spectral, num_fm=num_fm, reuse=False, weight_decay=weight_decay)
        Xs = X

        for _ in range(num_ite):
            X = boost_lap(X, Z, Y, num_spectral=num_spectral, num_fm=num_fm, reuse=True, weight_decay=weight_decay)
            Xs = tf.concat([Xs, X], axis=3)

        X = ly.conv2d(
            Xs,
            num_outputs=num_spectral,
            kernel_size=3,
            stride=1,
            weights_regularizer=ly.l2_regularizer(weight_decay),
            weights_initializer=ly.variance_scaling_initializer(),
            activation_fn=None,
        )
        return X


def main():
    parser = argparse.ArgumentParser(description='Python-only DBIN evaluation')
    parser.add_argument('data', type=str, help='Path to test TFRecord file OR a directory containing mat files (gt.mat, ms.mat, pan.mat, optional pan2/4, gt2/4, ms2)')
    parser.add_argument('--model_dir', type=str, default=os.path.join('methods', '_DBIN', 'models_ibp_sn22'), help='Checkpoint directory (default: methods/_DBIN/models_ibp_sn22)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_images', type=int, default=12, help='Number of test samples to evaluate')
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    args = parser.parse_args()

    # Basic TF1 check
    if tf.__version__.startswith('2'):
        raise RuntimeError('This script requires TensorFlow 1.x (tf.contrib). Please install TF 1.15.x.')

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    batch_size = args.batch_size
    image_size = args.image_size
    weight_decay = args.weight_decay

    # Placeholders
    gt_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 31])
    ms_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size // 8, image_size // 8, 31])
    pan_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
    pan2_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size // 2, image_size // 2, 3])
    pan4_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size // 4, image_size // 4, 3])

    # If "data" is a directory, auto-convert mats to a test TFRecord first
    data_path = args.data
    if os.path.isdir(data_path):
        out_dir = os.path.join('methods', '_DBIN', 'training_data')
        _ensure_dir(out_dir)
        tfrecord_path = os.path.join(out_dir, 'autotest.tfrecords')
        print('Converting MATs in {} -> {}'.format(data_path, tfrecord_path))
        convert_mats_to_test_tfrecord(data_path, tfrecord_path)
        data_path = tfrecord_path

    # Dataset pipeline (uses existing reader)
    pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch = read_and_decode_test(data_path, batch_size=batch_size)

    # Build model
    X = fusion_net(pan_holder, ms_holder, num_spectral=31, num_fm=128, num_ite=5, reuse=False, weight_decay=weight_decay)
    output = tf.clip_by_value(X, 0.0, 1.0)

    mse = tf.square(output - gt_holder)
    mse = tf.reshape(mse, [image_size, image_size, 31])
    final_mse = tf.reduce_mean(mse, axis=[0, 1])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    average_psnr = 0.0
    average_ssim = 0.0
    average_sam = 0.0
    average_ergas = 0.0

    gt_out = np.zeros(shape=[args.num_images, image_size, image_size, 31], dtype=np.float32)
    net_out = np.zeros(shape=[args.num_images, image_size, image_size, 31], dtype=np.float32)

    with tf.Session(config=config) as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Restore checkpoint
        if tf.train.get_checkpoint_state(args.model_dir):
            ckpt = tf.train.latest_checkpoint(args.model_dir)
            saver.restore(sess, ckpt)
            print('Loaded checkpoint:', ckpt)
        else:
            raise RuntimeError('No checkpoint found in {}'.format(args.model_dir))

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
            temp_ergas = compute_ergas(mse_loss, out)

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
