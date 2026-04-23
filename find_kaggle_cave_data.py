#!/usr/bin/env python3
"""
Diagnostic script to find available HSI/RGB data on Kaggle
"""

import os
import glob
from pathlib import Path

base_paths = [
    "/kaggle/input/cave-dataset-2",
    "/kaggle/input/cave-dataset",
    "/kaggle/input/CAVE",
    "/kaggle/input",
]

print("=" * 80)
print("Searching for CAVE dataset on Kaggle...")
print("=" * 80)

for base in base_paths:
    if os.path.exists(base):
        print(f"\n✓ Found: {base}")
        
        # Find HSI directories
        hsi_dirs = glob.glob(f"{base}/**/HSI", recursive=True)
        hsi_dirs += glob.glob(f"{base}/**/hsi", recursive=True)
        hsi_dirs += glob.glob(f"{base}/**/Data/Test", recursive=True)
        
        for hsi_dir in hsi_dirs[:5]:  # Limit output
            print(f"\n  HSI Directory: {hsi_dir}")
            files = os.listdir(hsi_dir)[:10]
            for f in files:
                print(f"    - {f}")
            total = len(os.listdir(hsi_dir))
            if total > 10:
                print(f"    ... and {total-10} more files")

print("\n" + "=" * 80)
print("Script: To use with test_kaggle_direct.py, run:")
print("python methods/_TSFN/test_kaggle_direct.py \\")
print("  --hsi_dir <HSI_PATH_FROM_ABOVE> \\")
print("  --rgb_dir <RGB_PATH_FROM_ABOVE> \\")
print("  --sf 8")
print("=" * 80)
