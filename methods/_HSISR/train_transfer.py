#!/usr/bin/env python
"""
Transfer learning: Fine-tune pre-trained SF=4 model on SF=8 dataset.
Much faster than training from scratch (30 epochs vs 100+).

Usage:
    python train_transfer.py \
        --pretrain_path pretrained_sf4.pth \
        --hsi_dir data/HSI \
        --rgb_dir data/MS \
        --sf 8 \
        --epochs 30 \
        --batch_size 16 \
        --gpus 2
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import tifffile
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/workspaces/hif-benchmarking/auxiliary/globals')
sys.path.insert(0, '/workspaces/hif-benchmarking/methods/_HSISR')

from DeepShare import DeepShare
from BlockModule import BlockModule


def normalize01(data):
    """Normalize to [0, 1]."""
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8)


def load_hsi_image(path):
    """Load HSI from TIF file."""
    return np.array(tifffile.imread(path)).astype(np.float32)


def load_rgb_image(path):
    """Load RGB from TIF file."""
    img = Image.open(path)
    return np.array(img).astype(np.float32)


def bicubic_downsample(img, sf):
    """Downsample using bicubic interpolation."""
    h, w = img.shape[:2]
    h_new, w_new = h // sf, w // sf
    if len(img.shape) == 3:
        return np.array(Image.fromarray(img.astype(np.uint8)).resize(
            (w_new, h_new), Image.BICUBIC)).astype(np.float32)
    else:
        return np.array(Image.fromarray(img.astype(np.uint8)).resize(
            (w_new, h_new), Image.BICUBIC)).astype(np.float32)


class HSIPatchDataset(Dataset):
    """HSI patch dataset for training."""
    
    def __init__(self, hsi_dir, rgb_dir, sf, patch_size=64, stride=32):
        self.sf = sf
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        
        hsi_files = sorted([f for f in os.listdir(hsi_dir) if f.endswith('.tif')])
        
        print(f"Loading {len(hsi_files)} HSI images...")
        for hsi_file in hsi_files:
            rgb_file = hsi_file.replace('.tif', '.tif')
            hsi_path = os.path.join(hsi_dir, hsi_file)
            rgb_path = os.path.join(rgb_dir, rgb_file)
            
            if not os.path.exists(rgb_path):
                continue
            
            hsi = load_hsi_image(hsi_path)
            rgb = load_rgb_image(rgb_path)
            
            h, w = hsi.shape[:2]
            h_new = (h // sf) * sf
            w_new = (w // sf) * sf
            
            hsi = hsi[:h_new, :w_new]
            rgb = rgb[:h_new, :w_new]
            
            hsi = normalize01(hsi)
            rgb = normalize01(rgb)
            
            # Extract patches
            for i in range(0, h_new - patch_size + 1, stride):
                for j in range(0, w_new - patch_size + 1, stride):
                    hsi_patch = hsi[i:i+patch_size, j:j+patch_size]
                    rgb_patch = rgb[i:i+patch_size, j:j+patch_size]
                    
                    # Downsample for LR-HSI
                    lrhsi = bicubic_downsample(hsi_patch, sf)
                    rgb_lr = np.array(Image.fromarray((rgb_patch*255).astype(np.uint8)).resize(
                        (patch_size//sf, patch_size//sf), Image.BICUBIC)) / 255.0
                    rgb_up = np.array(Image.fromarray((rgb_lr*255).astype(np.uint8)).resize(
                        (patch_size, patch_size), Image.BICUBIC)) / 255.0
                    
                    self.patches.append({
                        'hsi': hsi_patch.transpose(2, 0, 1),
                        'lrhsi': lrhsi.transpose(2, 0, 1),
                        'rgb': rgb_up.transpose(2, 0, 1)
                    })
            
            print(f"  {hsi_file}: {len(self.patches)} patches so far")
        
        print(f"Total patches: {len(self.patches)}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        # Random flip
        if np.random.rand() > 0.5:
            patch['hsi'] = patch['hsi'][:, :, ::-1].copy()
            patch['lrhsi'] = patch['lrhsi'][:, :, ::-1].copy()
            patch['rgb'] = patch['rgb'][:, :, ::-1].copy()
        
        return {
            'hsi': torch.from_numpy(patch['hsi']).float(),
            'lrhsi': torch.from_numpy(patch['lrhsi']).float(),
            'rgb': torch.from_numpy(patch['rgb']).float()
        }


def train_epoch(model, loader, optimizer, device):
    """Train one epoch."""
    model.train()
    loss_meter = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        hsi = batch['hsi'].to(device)
        lrhsi = batch['lrhsi'].to(device)
        rgb = batch['rgb'].to(device)
        
        optimizer.zero_grad()
        
        pred = model(lrhsi, rgb)
        loss = nn.MSELoss()(pred, hsi)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_meter += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return loss_meter / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', type=str, required=True, help='Path to pretrained SF=4 model')
    parser.add_argument('--hsi_dir', type=str, required=True, help='HSI training directory')
    parser.add_argument('--rgb_dir', type=str, required=True, help='RGB training directory')
    parser.add_argument('--sf', type=int, default=8, help='Scale factor')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (lower for transfer)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--save_dir', type=str, default='checkpoints_sf8_transfer', help='Save directory')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print("\n📊 Loading training dataset...")
    dataset = HSIPatchDataset(args.hsi_dir, args.rgb_dir, args.sf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    print(f"\n🏗️  Loading pretrained model from: {args.pretrain_path}")
    model = DeepShare(BlockModule, n_subs=8, n_ovls=2, n_feats=256, n_blocks=3)
    
    checkpoint = torch.load(args.pretrain_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    if args.gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.gpus)))
    model = model.to(device)
    
    # Optimizer (lower LR for transfer learning)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\n🚀 Starting transfer learning (SF={args.sf}, LR={args.lr})")
    print(f"   Epochs: {args.epochs}, Batch size: {args.batch_size}, GPUs: {args.gpus}")
    print()
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        
        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(args.save_dir, 'BEST.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}/{args.epochs}: Loss={loss:.6f} ↓ ⭐ NEW BEST!")
        else:
            print(f"Epoch {epoch}/{args.epochs}: Loss={loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete! Total time: {elapsed/60:.1f} min")
    print(f"   Best model saved to: checkpoints_sf8_transfer/BEST.pth")


if __name__ == '__main__':
    main()
