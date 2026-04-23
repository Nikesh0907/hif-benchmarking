#!/usr/bin/env python3
"""
HSISR training for CAVE dataset - Clean single file

Training time: ~1-2 hours per 50 epochs on V100 GPU

Usage:
  python train.py --hsi_dir /path/to/hsi --rgb_dir /path/to/rgb --sf 8 --epochs 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import glob
from PIL import Image
from tqdm import tqdm
import time

from BlockModule import DeepShare
from basicModule import default_conv
from Loss import HybridLoss


def load_mat(path):
    """Load .mat file with auto-detection."""
    mat_data = sio.loadmat(path)
    for key in ['hsi', 'gt', 'msi', 'X']:
        if key in mat_data:
            arr = mat_data[key]
            break
    else:
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        arr = mat_data[keys[0]] if keys else None
    
    if arr is None:
        return None
    
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = arr.transpose(1, 2, 0)
    
    arr_max = float(np.nanmax(arr)) if arr.size else 1.0
    if arr_max > 1.0:
        arr = arr / arr_max
    
    return np.clip(arr, 0.0, 1.0)


def downsample(img, sf):
    """Bicubic downsample using PIL."""
    h, w, c = img.shape
    h_lr, w_lr = h // sf, w // sf
    
    img_lr = np.zeros((h_lr, w_lr, c), dtype=np.float32)
    for i in range(c):
        pil_img = Image.fromarray((img[:, :, i] * 255).astype(np.uint8))
        pil_ds = pil_img.resize((w_lr, h_lr), Image.BICUBIC)
        img_lr[:, :, i] = np.array(pil_ds, dtype=np.float32) / 255.0
    return img_lr


def upsample(img_lr, sf):
    """Bicubic upsample using PIL."""
    h, w, c = img_lr.shape
    h_hr, w_hr = h * sf, w * sf
    
    img_up = np.zeros((h_hr, w_hr, c), dtype=np.float32)
    for i in range(c):
        pil_img = Image.fromarray((img_lr[:, :, i] * 255).astype(np.uint8))
        pil_up = pil_img.resize((w_hr, h_hr), Image.BICUBIC)
        img_up[:, :, i] = np.array(pil_up, dtype=np.float32) / 255.0
    return img_up


class HSIPatchDataset(torch.utils.data.Dataset):
    """Load HSI+RGB pairs and create training patches."""
    
    def __init__(self, hsi_files, rgb_files, sf=8, patch_size=64):
        self.hsi_files = hsi_files
        self.rgb_files = rgb_files
        self.sf = sf
        self.patch_size = patch_size
        self.patches = []
        
        print(f"Loading {len(hsi_files)} images...")
        for hsi_path, rgb_path in zip(hsi_files, rgb_files):
            try:
                hsi = load_mat(hsi_path)
                rgb = load_mat(rgb_path) if rgb_path else None
                
                if hsi is None:
                    continue
                
                if hsi.shape[2] > 31:
                    hsi = hsi[:, :, :31]
                
                h, w = hsi.shape[:2]
                h = (h // sf) * sf
                w = (w // sf) * sf
                hsi = hsi[:h, :w, :]
                
                for i in range(0, h - patch_size + 1, 32):
                    for j in range(0, w - patch_size + 1, 32):
                        hsi_patch = hsi[i:i+patch_size, j:j+patch_size, :]
                        hsi_lr = downsample(hsi_patch, sf)
                        hsi_lr_up = upsample(hsi_lr, sf)
                        
                        if rgb is not None:
                            rgb_patch = rgb[i:i+patch_size, j:j+patch_size, :]
                            if rgb_patch.shape[2] > 3:
                                rgb_patch = rgb_patch[:, :, :3]
                        else:
                            idx_r, idx_g, idx_b = 23, 15, 7
                            rgb_patch = np.stack([
                                hsi_patch[..., idx_r],
                                hsi_patch[..., idx_g],
                                hsi_patch[..., idx_b]
                            ], axis=-1)
                        
                        self.patches.append({
                            'gt': hsi_patch,
                            'lr_hsi': hsi_lr,
                            'lr_hsi_up': hsi_lr_up,
                            'rgb': rgb_patch
                        })
                
                print(f"  ✓ {os.path.basename(hsi_path)}: {(h//patch_size)*(w//patch_size)} patches")
            except Exception as e:
                print(f"  ✗ {os.path.basename(hsi_path)}: {e}")
        
        print(f"Total patches: {len(self.patches)}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        if np.random.rand() > 0.5:
            patch['gt'] = np.flipud(patch['gt'])
            patch['lr_hsi'] = np.flipud(patch['lr_hsi'])
            patch['lr_hsi_up'] = np.flipud(patch['lr_hsi_up'])
            patch['rgb'] = np.flipud(patch['rgb'])
        
        if np.random.rand() > 0.5:
            patch['gt'] = np.fliplr(patch['gt'])
            patch['lr_hsi'] = np.fliplr(patch['lr_hsi'])
            patch['lr_hsi_up'] = np.fliplr(patch['lr_hsi_up'])
            patch['rgb'] = np.fliplr(patch['rgb'])
        
        gt = torch.from_numpy(patch['gt'].transpose(2, 0, 1))
        lr_hsi = torch.from_numpy(patch['lr_hsi'].transpose(2, 0, 1))
        lr_hsi_up = torch.from_numpy(patch['lr_hsi_up'].transpose(2, 0, 1))
        rgb = torch.from_numpy(patch['rgb'].transpose(2, 0, 1))
        
        return lr_hsi, lr_hsi_up, gt, rgb


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for lr_hsi, lr_hsi_up, gt, rgb in pbar:
        lr_hsi = lr_hsi.to(device)
        lr_hsi_up = lr_hsi_up.to(device)
        gt = gt.to(device)
        rgb = rgb.to(device)
        
        output = model(lr_hsi, lr_hsi_up, modality="spectral")
        loss = criterion(output, gt)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_val = loss.item()
        pbar.set_postfix({'loss': loss_val})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train HSISR on CAVE')
    parser.add_argument('--hsi_dir', type=str, required=True, help='HSI .mat directory')
    parser.add_argument('--rgb_dir', type=str, default=None, help='RGB .mat directory')
    parser.add_argument('--sf', type=int, default=8, help='Scale factor')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save dir')
    parser.add_argument('--cuda', type=int, default=1, help='Use CUDA')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"HSISR Training (SF={args.sf}, Epochs={args.epochs})")
    print("=" * 70)
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    hsi_files = sorted(glob.glob(os.path.join(args.hsi_dir, '*.mat')))
    if not hsi_files:
        raise ValueError(f"No .mat files in {args.hsi_dir}")
    
    rgb_files = None
    if args.rgb_dir and os.path.exists(args.rgb_dir):
        rgb_files = [os.path.join(args.rgb_dir, os.path.basename(f)) for f in hsi_files]
        rgb_files = [f if os.path.exists(f) else None for f in rgb_files]
    
    print(f"HSI files: {len(hsi_files)}")
    print(f"RGB files: {len([f for f in rgb_files if f]) if rgb_files else 'None'}\n")
    
    print("[1/3] Creating dataset...")
    dataset = HSIPatchDataset(hsi_files, rgb_files or [None]*len(hsi_files), 
                              sf=args.sf, patch_size=args.patch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=4)
    print(f"Batches per epoch: {len(dataloader)}\n")
    
    print("[2/3] Building model...")
    model = DeepShare(n_subs=8, n_ovls=2, n_colors=31, n_blocks=3, n_feats=256,
                      n_scale=args.sf, res_scale=0.1, use_share=True, conv=default_conv)
    model = model.to(device)
    print(f"Model: DeepShare(sf={args.sf}, feats=256, blocks=3)\n")
    
    criterion = HybridLoss(spatial_tv=True, spectral_tv=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print("[3/3] Training...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}\n")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = (time.time() - start_time) / 60
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs}: Loss={avg_loss:.6f}, LR={current_lr:.2e} ({elapsed:.1f}min, ETA {eta:.1f}min)")
        
        if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
            ckpt_path = os.path.join(args.save_dir, f'CAVE_DeepShare_SF{args.sf}_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved: {ckpt_path}")
            if avg_loss < best_loss:
                best_loss = avg_loss
    
    total_time = (time.time() - start_time) / 3600
    print("\n" + "=" * 70)
    print(f"Training complete! Time: {total_time:.2f} hours")
    print(f"Best checkpoint: {args.save_dir}/CAVE_DeepShare_SF{args.sf}_epoch*.pth")
    print("=" * 70)


if __name__ == '__main__':
    main()
