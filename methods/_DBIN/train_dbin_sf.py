#!/usr/bin/env python3

import argparse
import os

import numpy as np


def read_and_decode_train_sf(tfrecords_file, batch_size, patch_size, sf):
    import tensorflow as tf

    if sf < 2 or (sf % 2) != 0:
        raise ValueError('sf must be an even integer >= 2')
    if patch_size % sf != 0:
        raise ValueError('patch_size must be divisible by sf')

    ms_size = patch_size // sf

    filename_queue = tf.compat.v1.train.string_input_producer([tfrecords_file])
    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_features = tf.compat.v1.parse_single_example(
        serialized_example,
        features={
            'pan_raw': tf.io.FixedLenFeature([], tf.string),
            'gt_raw': tf.io.FixedLenFeature([], tf.string),
            'ms_raw': tf.io.FixedLenFeature([], tf.string),
        },
    )

    pan = tf.io.decode_raw(img_features['pan_raw'], tf.float32)
    pan = tf.reshape(pan, [patch_size, patch_size, 3])
    gt = tf.io.decode_raw(img_features['gt_raw'], tf.float32)
    gt = tf.reshape(gt, [patch_size, patch_size, 31])
    ms = tf.io.decode_raw(img_features['ms_raw'], tf.float32)
    ms = tf.reshape(ms, [ms_size, ms_size, 31])

    pan_b, gt_b, ms_b = tf.compat.v1.train.shuffle_batch(
        [pan, gt, ms],
        batch_size=batch_size,
        capacity=300,
        min_after_dequeue=200,
        num_threads=4,
        allow_smaller_final_batch=False,
    )
    return pan_b, gt_b, ms_b


def main():
    ap = argparse.ArgumentParser(description='Train DBIN on arbitrary scale factors (sf)')
    ap.add_argument('--train_tfrecord', required=True, help='Training TFRecord created by make_train_tfrecord.py')
    ap.add_argument('--model_dir', required=True, help='Output checkpoint directory')
    ap.add_argument('--sf', type=int, default=8, help='Scale factor (even >=2). e.g., 8/16/32')
    ap.add_argument('--patch', type=int, default=64, help='Patch size (divisible by sf).')
    ap.add_argument('--batch', type=int, default=32, help='Batch size')
    ap.add_argument('--iters', type=int, default=250000, help='Training iterations')
    ap.add_argument('--save_every', type=int, default=10000, help='Checkpoint save interval')
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (data-parallel towers)')
    ap.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint in model_dir')
    args = ap.parse_args()

    sf = int(args.sf)
    if sf < 2 or (sf % 2) != 0:
        raise SystemExit('sf must be even and >=2')
    if args.patch % sf != 0:
        raise SystemExit('patch must be divisible by sf')

    import tensorflow as tf
    if tf.__version__.startswith('2'):
        tf.compat.v1.disable_eager_execution()

    tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Placeholders
    gt_ph = tf.compat.v1.placeholder(tf.float32, [args.batch, args.patch, args.patch, 31])
    pan_ph = tf.compat.v1.placeholder(tf.float32, [args.batch, args.patch, args.patch, 3])
    ms_ph = tf.compat.v1.placeholder(tf.float32, [args.batch, args.patch // sf, args.patch // sf, 31])
    lr_ph = tf.compat.v1.placeholder(tf.float32, [])
    global_step = tf.compat.v1.Variable(0, trainable=False, name='global_step')

    pan_b, gt_b, ms_b = read_and_decode_train_sf(args.train_tfrecord, args.batch, args.patch, sf)

    # Reuse the model definition from dbintest.py (same variable naming)
    from methods._DBIN.dbintest import fusion_net

    # Multi-GPU towers: split batch across GPUs
    num_gpus = max(1, int(args.gpus))
    pan_slices = tf.split(pan_ph, num_gpus, axis=0)
    gt_slices = tf.split(gt_ph, num_gpus, axis=0)
    ms_slices = tf.split(ms_ph, num_gpus, axis=0)

    opt = tf.compat.v1.train.AdamOptimizer(lr_ph, beta1=0.9)
    tower_grads = []
    tower_losses = []

    for gi in range(num_gpus):
        dev = '/gpu:%d' % gi if tf.config.list_physical_devices('GPU') else '/cpu:0'
        with tf.device(dev):
            pred_i = fusion_net(
                pan_slices[gi],
                ms_slices[gi],
                num_spectral=31,
                num_fm=64,
                num_ite=8,
                sf=sf,
                reuse=(gi > 0),
                weight_decay=args.weight_decay,
            )
            loss_i = tf.reduce_mean(tf.abs(pred_i - gt_slices[gi]))
            tower_losses.append(loss_i)
            # Ensure variables are collected once
            t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='fusion_net')
            grads_i = opt.compute_gradients(loss_i, var_list=t_vars)
            tower_grads.append(grads_i)

    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = [g for g, v in grad_and_vars if g is not None]
            if not grads:
                average_grads.append((None, grad_and_vars[0][1]))
                continue
            grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)
            v = grad_and_vars[0][1]
            average_grads.append((grad, v))
        return average_grads

    avg_grads = average_gradients(tower_grads)
    loss = tf.reduce_mean(tf.stack(tower_losses, axis=0))
    train_op = opt.apply_gradients(avg_grads, global_step=global_step)

    init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    os.makedirs(args.model_dir, exist_ok=True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # Optionally resume
        start_step = 0
        if args.resume:
            ckpt = tf.compat.v1.train.get_checkpoint_state(args.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                start_step = int(sess.run(global_step))
                print('Resumed from', ckpt.model_checkpoint_path, 'starting at step', start_step)
            else:
                print('No checkpoint found in', args.model_dir, '; starting fresh.')

        for i in range(start_step, int(args.iters) + 1):
            # Same LR schedule style as original scripts
            if i <= 20000:
                lr = 4e-4
            elif i <= 60000:
                lr = 2e-4
            elif i <= 140000:
                lr = 1e-4
            else:
                lr = 5e-5

            pan_np, gt_np, ms_np = sess.run([pan_b, gt_b, ms_b])
            _, l, step_val = sess.run(
                [train_op, loss],
                feed_dict={pan_ph: pan_np, gt_ph: gt_np, ms_ph: ms_np, lr_ph: lr},
            )

            if i % 100 == 0:
                print(f'Iter {i} step {step_val} loss {l:.6f} lr {lr:g}')

            if (i % int(args.save_every) == 0 and i != 0) or i == int(args.iters):
                saver.save(sess, os.path.join(args.model_dir, f'model-{i}.ckpt'))
                print('Saved checkpoint')

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
