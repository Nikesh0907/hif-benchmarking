#!/usr/bin/env python3
"""Diagnose CAVE HSI data format and range."""

import scipy.io as sio
import numpy as np
import os
import sys

test_hsi_dir = '/kaggle/input/cave-dataset-2/Data/Test/HSI'

if not os.path.isdir(test_hsi_dir):
    print(f"ERROR: {test_hsi_dir} not found")
    print("Please update the path to your CAVE test HSI directory")
    sys.exit(1)

print("=" * 70)
print(f"Diagnosing CAVE HSI files in: {test_hsi_dir}")
print("=" * 70)

mat_files = sorted([f for f in os.listdir(test_hsi_dir) if f.endswith('.mat')])[:3]

for fname in mat_files:
    fpath = os.path.join(test_hsi_dir, fname)
    mat = sio.loadmat(fpath)
    
    print(f"\n{fname}:")
    print(f"  Keys: {[k for k in mat.keys() if not k.startswith('__')]}")
    
    # Find the HSI data (skip MATLAB metadata)
    for k in mat.keys():
        if not k.startswith('__'):
            arr = np.asarray(mat[k], dtype=np.float32)
            print(f"    {k}:")
            print(f"      Shape: {arr.shape}")
            print(f"      Dtype: {arr.dtype}")
            print(f"      Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f}")
            print(f"      Mean: {np.mean(arr):.4f}")
            
            # Guess the normalization
            mx = np.max(arr)
            if mx > 255:
                print(f"      → Likely range: [0, {int(mx)}] (NOT normalized to [0,1])")
            elif mx > 1:
                print(f"      → Likely range: [0, 255] or [0, 4095] converted to float")
            else:
                print(f"      → Already normalized to [0, 1]")

print("\n" + "=" * 70)
