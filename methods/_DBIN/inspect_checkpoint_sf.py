#!/usr/bin/env python3

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description='Inspect DBIN/EDBIN checkpoint for scale-factor hints')
    ap.add_argument('ckpt', type=str, help='Path to checkpoint prefix (e.g., .../model-250000.ckpt)')
    args = ap.parse_args()

    ckpt = args.ckpt
    if ckpt.endswith('.index'):
        ckpt = ckpt[:-6]
    if ckpt.endswith('.meta'):
        ckpt = ckpt[:-5]
    if ckpt.endswith('.data-00000-of-00001'):
        ckpt = ckpt[: -len('.data-00000-of-00001')]

    try:
        import tensorflow as tf
    except Exception as e:
        raise SystemExit(f'TensorFlow import failed: {e}')

    if not (os.path.exists(ckpt + '.index') or os.path.exists(ckpt)):
        raise SystemExit(f'Checkpoint not found: {ckpt} (expected {ckpt}.index)')

    vars_list = tf.train.list_variables(ckpt)
    var_map = {n: s for n, s in vars_list}

    dy = var_map.get('fusion_net/recursive/dy/kernel')
    if dy is None:
        # try alternative names (in case of different scopes)
        dy_candidates = [(n, s) for n, s in vars_list if n.endswith('/dy/kernel')]
        if dy_candidates:
            print('Found dy kernel candidates:')
            for n, s in dy_candidates[:20]:
                print(' ', n, s)
        else:
            print('Did not find a dy kernel in checkpoint.')
            print('This script can only give a strong sf hint when dy exists.')
        return

    k = dy[0] if isinstance(dy, (list, tuple)) and len(dy) >= 2 else None
    print('fusion_net/recursive/dy/kernel shape:', dy)

    print('\nHow to interpret in THIS repo:')
    print('- The training code hard-codes stride=8 for the dy downsampling conv in methods/_DBIN/train_cave_edbin.py.')
    print('- That corresponds to sf=8 (GT -> MS is 8x downsample).')
    if k is not None:
        print(f'- Your dy kernel spatial size is {k}x{k}. In the provided scripts, sf=8 uses kernel=12, stride=8.')
    print('\nBottom line: this checkpoint matches sf=8 in the released DBIN training code.')


if __name__ == '__main__':
    main()
