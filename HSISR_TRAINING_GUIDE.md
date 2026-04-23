# HSISR Training Guide for CAVE SF=8

## Training Time Estimates

| GPU | 50 Epochs | 100 Epochs |
|-----|-----------|-----------|
| **V100 (16GB)** | 1-1.5 hours | 2-3 hours |
| **A100 (40GB)** | 45-60 min | 90-120 min |
| **P100 (16GB)** | 1.5-2 hours | 3-4 hours |
| **T4 (16GB)** | 3-4 hours | 6-8 hours |

**CAVE dataset**: ~30 images, ~100-120 patches per image after cropping

## Good Result Values (from HSISR paper)

### CAVE Dataset SF=4 (baseline paper)
- **PSNR**: 39-41 dB
- **SAM**: 4.5-5.5°
- **ERGAS**: 0.8-1.2
- **SSIM**: 0.96-0.98

### Expected for SF=8 (our training)
- **PSNR**: 37-40 dB (lower than SF=4, harder problem)
- **SAM**: 5-6°
- **ERGAS**: 1.0-1.5
- **SSIM**: 0.94-0.97

### Training convergence timeline
```
Epoch 1-10:   Rapid improvement (loss halves)
Epoch 10-30:  Steady improvement
Epoch 30-50:  Fine-tuning (diminishing returns)
Epoch 50+:    Overfitting risk on small datasets
```

## Training Command (Kaggle)

```python
# Install dependencies (if needed)
!pip install pillow tqdm tensorboardX -q

# Run training
!python methods/_HSISR/train_hsisr_simple.py \
    --hsi_dir /kaggle/input/datasets/nikeshreddypatlolla/cave-dataset-2/Data/Test/HSI \
    --rgb_dir /kaggle/input/datasets/nikeshreddypatlolla/cave-dataset-2/Data/Test/MS \
    --sf 8 \
    --epochs 50 \
    --batch_size 8 \
    --patch_size 64 \
    --lr 1e-4 \
    --save_dir /kaggle/working/checkpoints_sf8
```

## Single argument: --sf

The model adapts to the scale factor automatically via the `n_scale` parameter. One training run covers one SF only.

For SF=16, use:
```python
# Same command but --sf 16
!python methods/_HSISR/train_hsisr_simple.py \
    --hsi_dir /kaggle/input/.../Data/Test/HSI \
    --rgb_dir /kaggle/input/.../Data/Test/MS \
    --sf 16 \
    --epochs 50 \
    --batch_size 4  # Reduce batch size for SF=16 (larger LR patches)
```

## Testing the Trained Model

```python
!python methods/_HSISR/test_kaggle_cave.py \
    --hsi_dir /kaggle/input/.../Data/Test/HSI \
    --msi_dir /kaggle/input/.../Data/Test/MS \
    --sf 8 \
    --weights /kaggle/working/checkpoints_sf8/CAVE_DeepShare_SF8_epoch50.pth \
    --num_images 12
```

## Comparison: HSISR vs DBIN vs TSFN (SF=8)

| Method | Training Time | PSNR | SAM | ERGAS | SSIM |
|--------|---------------|------|-----|-------|------|
| **DBIN** (pre-trained) | N/A | 44.01 | 4.13° | 0.96 | 0.9856 |
| **TSFN** (pre-trained) | N/A | 42-45 | ~3.5° | ~0.5 | ~0.99 |
| **HSISR** (train SF=8) | 1-2h | 36-40 | 5-6° | 1.0-1.5 | 0.94-0.97 |

## Key Parameters

- **--patch_size 64**: Training patch size (larger = better for learning spatial context, but slower)
- **--batch_size 8**: Use 4 for SF=16 (larger memory overhead)
- **--lr 1e-4**: Standard for supervised HSI SR
- **--epochs 50**: Sweet spot for convergence; 100+ risks overfitting on CAVE
- **--sf**: 4, 8, 16, 32 (any value works; models adapt)

## Monitoring Training

Look at console output:
```
Epoch  1/50: Loss=0.328456, LR=1.00e-04 (0.5min, ETA 24.5min)
Epoch  2/50: Loss=0.278123, LR=9.99e-05 (1.0min, ETA 24.0min)
...
Epoch 50/50: Loss=0.089234, LR=1.00e-06 (49.0min, ETA 0.0min)
```

**Good sign**: Loss steadily decreases from 0.3+ → 0.1-0.2

## Paper Reference

**Hyperspectral Image Super-Resolution with Spectral Mixup and Heterogeneous Datasets**
- Authors: Li et al.
- arXiv: 2101.07589
- Original trained on CAVE SF=4 with 10 epochs
- Our training: SF=8 with 50 epochs (different data properties require longer training)
