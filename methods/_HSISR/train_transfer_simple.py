#!/usr/bin/env python
"""
Quick wrapper to train with pretrained weights using existing train.py
"""
import sys
import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', type=str, required=True)
parser.add_argument('--hsi_dir', type=str, required=True)
parser.add_argument('--rgb_dir', type=str, required=True)
parser.add_argument('--sf', type=int, default=8)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='checkpoints_sf8_transfer')

args = parser.parse_args()

# Build train.py command
train_args = [
    'train.py',
    '--hsi_dir', args.hsi_dir,
    '--rgb_dir', args.rgb_dir,
    '--sf', str(args.sf),
    '--epochs', str(args.epochs),
    '--batch_size', str(args.batch_size),
    '--gpus', str(args.gpus),
    '--save_dir', args.save_dir,
    '--pretrain_path', args.pretrain_path,
    '--lr', '1e-5'
]

sys.argv = train_args
sys.path.insert(0, '/workspaces/hif-benchmarking/methods/_HSISR')

# Import and run train.py
from train import main
main()
