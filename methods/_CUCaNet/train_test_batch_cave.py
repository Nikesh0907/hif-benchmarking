#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch train + test CUCaNet on multiple CAVE HSI files.

Usage (Kaggle):
  python methods/_CUCaNet/train_test_batch_cave.py \
    --train_dir /path/to/train/hsi/mats \
    --test_dir /path/to/test/hsi/mats \
    --srf_name Nikon_D700_Qu \
    --scale_factor 32 \
    --niter 2000 \
    --niter_decay 8000
"""

import argparse
import os
import sys
import glob
import time
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

# Add CUCaNet to path
cucadir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cucadir)

from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_and_test_single(train_mat_path, test_mat_paths, train_opt, cave_dir="./CAVE"):
    """Train CUCaNet on a single HSI file and test on multiple HSIs."""
    mat_name = Path(train_mat_path).stem
    print(f"\n{'='*80}")
    print(f"Training on: {mat_name}")
    print(f"{'='*80}")
    
    # Copy training file to CAVE folder (CUCaNet expects it there)
    os.makedirs(cave_dir, exist_ok=True)
    cave_mat_path = os.path.join(cave_dir, mat_name + ".mat")
    print(f"Copying {train_mat_path} to {cave_mat_path}")
    shutil.copy(train_mat_path, cave_mat_path)
    
    # Ensure the mat file has key 'img' (rename 'hsi' if needed)
    mat_data = sio.loadmat(cave_mat_path)
    if 'img' not in mat_data and 'hsi' in mat_data:
        mat_data['img'] = mat_data.pop('hsi')
        sio.savemat(cave_mat_path, mat_data)
        print(f"Renamed 'hsi' key to 'img' in {cave_mat_path}")
    
    train_opt.name = f"CAVE_{mat_name}"
    train_opt.mat_name = mat_name
    
    # Create directories
    checkpoint_dir = os.path.join(train_opt.checkpoints_dir, train_opt.name)
    results_dir = os.path.join(train_opt.results_dir, "CAVE")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup data loader for training
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    
    # Create model
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)
    
    train_model.setup(train_opt)
    
    # Train
    print(f"Training for {train_opt.niter + train_opt.niter_decay} epochs...")
    total_steps = 0
    for epoch in range(1, train_opt.niter + train_opt.niter_decay + 1):
        for i, data in enumerate(train_dataloader):
            total_steps += 1
            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)
            
            if epoch % train_opt.print_freq == 0:
                psnr = train_model.cal_psnr()
                print(f"Epoch {epoch}/{train_opt.niter + train_opt.niter_decay}, PSNR: {psnr:.4f}")
        
        if epoch % train_opt.save_freq == 0:
            train_model.save_networks(epoch)
        
        train_model.update_learning_rate()
    
    print(f"Training complete. Checkpoint saved to {checkpoint_dir}")
    
    # Test on multiple files
    print(f"\nTesting on {len(test_mat_paths)} test files...")
    test_results = {}
    for test_mat_path in test_mat_paths:
        test_name = Path(test_mat_path).stem
        print(f"  Testing {test_name}...", end=" ")
        
        # Temporarily load test data
        test_mat = sio.loadmat(test_mat_path)
        if 'img' in test_mat:
            test_img = test_mat['img'].astype(np.float32)
        elif 'hsi' in test_mat:
            test_img = test_mat['hsi'].astype(np.float32)
        else:
            print("SKIP (no 'img' or 'hsi' key)")
            continue
        
        # Normalize to [0,1]
        if test_img.max() > 1.0:
            test_img = test_img / test_img.max()
        
        # Create test data dict (mock)
        test_data = {
            'img': torch.from_numpy(test_img[np.newaxis, ...]).permute(0, 3, 1, 2),
            'name': [test_name]
        }
        
        train_model.set_input(test_data, False)
        train_model.test()
        
        visuals = train_model.get_current_visuals()
        key_name = train_model.get_visual_corresponding_name()['real_hhsi']
        rec_hsi = visuals[key_name].data.cpu().float().numpy()[0]  # (C,H,W)
        rec_hsi = rec_hsi.transpose(1, 2, 0)  # (H,W,C)
        
        # Save result
        out_path = os.path.join(results_dir, f"{test_name}_trained_on_{mat_name}.mat")
        sio.savemat(out_path, {'out': rec_hsi})
        test_results[test_name] = out_path
        print(f"saved to {os.path.basename(out_path)}")
    
    return test_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Directory with training HSI .mat files")
    ap.add_argument("--test_dir", required=True, help="Directory with test HSI .mat files")
    ap.add_argument("--srf_name", default="Nikon_D700_Qu", help="Spectral response function name")
    ap.add_argument("--scale_factor", type=int, default=32, help="Scale factor")
    ap.add_argument("--niter", type=int, default=2000, help="Number of training iterations")
    ap.add_argument("--niter_decay", type=int, default=8000, help="Number of decay iterations")
    ap.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    ap.add_argument("--batchsize", type=int, default=1, help="Batch size (keep 1 for CUCaNet)")
    ap.add_argument("--max_train", type=int, default=999, help="Max number of training files to use")
    ap.add_argument("--max_test", type=int, default=999, help="Max number of test files to use")
    args = ap.parse_args()
    
    # Get file lists
    train_mats = sorted(glob.glob(os.path.join(args.train_dir, "*.mat")))[:args.max_train]
    test_mats = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))[:args.max_test]
    
    if not train_mats:
        raise SystemExit(f"No training .mat files found in {args.train_dir}")
    if not test_mats:
        raise SystemExit(f"No test .mat files found in {args.test_dir}")
    
    print(f"Found {len(train_mats)} training files and {len(test_mats)} test files")
    
    # Setup training options (use minimal sys.argv, then override)
    import sys as _sys
    old_argv = _sys.argv.copy()
    _sys.argv = [_sys.argv[0]]
    train_opt = TrainOptions().parse()
    _sys.argv = old_argv
    
    # Override with user args
    train_opt.data_name = "CAVE"
    train_opt.srf_name = args.srf_name
    train_opt.scale_factor = args.scale_factor
    train_opt.niter = args.niter
    train_opt.niter_decay = args.niter_decay
    train_opt.lr = args.lr
    train_opt.batchsize = args.batchsize
    train_opt.print_freq = max(1, train_opt.niter // 10)
    train_opt.save_freq = max(1, train_opt.niter // 5)
    train_opt.checkpoints_dir = "./checkpoints"
    train_opt.results_dir = "./Results"
    
    # Train and test on each training file
    all_results = {}
    for i, train_mat in enumerate(train_mats, start=1):
        print(f"\n[{i}/{len(train_mats)}]")
        results = train_and_test_single(train_mat, test_mats, train_opt, cave_dir="./CAVE")
        all_results[Path(train_mat).stem] = results
    
    print(f"\n{'='*80}")
    print("Batch training + testing complete!")
    print(f"Results saved to ./Results/CAVE/")
    print(f"{'='*80}")


if __name__ == "__main__":
    setup_seed(1)
    main()
