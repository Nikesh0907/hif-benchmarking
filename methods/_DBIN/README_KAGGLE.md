# DBIN (EDBIN checkpoint) – Kaggle Quickstart

This repo’s original DBIN/EDBIN testing code is TF1 + `tf.contrib` and does **not** run directly in Kaggle’s TF2/Keras 3 environment.

Use the Python-only evaluator:
- `methods/_DBIN/dbintest.py`

It:
- Accepts either a directory of per-scene `.mat` files (CAVE-style HSI) **or** a `.tfrecords` file
- Auto-converts MATs → TFRecord in `methods/_DBIN/training_data/autotest.tfrecords`
- Restores the provided checkpoint from `methods/_DBIN/models_ibp_sn22`
- Prints per-image metrics + averages and writes `result/out.mat`

## Kaggle notebook cells

### 1) Clone repo

```bash
!rm -rf hif-benchmarking
!git clone --depth 1 https://github.com/magamig/hif-benchmarking.git
%cd hif-benchmarking
```

### 2) Install deps

Kaggle usually already has most of these; this is safe if missing.

```bash
!pip -q install --upgrade "scikit-image" "opencv-python-headless" "scipy" "numpy"
```

### 3) Run DBIN evaluation

For good metrics, DBIN expects:
- **GT HSI** (31 bands) per scene
- **RGB/MSI** per scene (same basename)

Example (CAVE Kaggle datasets often have these folders):
- HSI: `/kaggle/input/cave-dataset-2/Data/Test/HSI`
- RGB: `/kaggle/input/cave-dataset-2/Data/Test/RGB` (or sometimes `MSI`)

Run:

```bash
!python methods/_DBIN/dbintest.py /kaggle/input/cave-dataset-2/Data/Test/HSI \
  --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
  --model_dir methods/_DBIN/models_ibp_sn22 \
  --batch_size 1 \
  --image_size 512 \
  --num_images 0
```

Notes:
- `--num_images 0` auto-counts records and avoids hanging if you guess wrong.
- Metrics are printed per image and averaged at the end.
- Outputs are saved to `result/out.mat`.

If you don't provide `--rgb_dir`, the script will synthesize a 3-channel proxy from 3 HSI bands. That usually produces **very poor PSNR/SSIM**, because it doesn't match the model's training input distribution.

### 4) Where outputs go

```bash
!ls -la result
```

## Common issues

- Lots of CUDA warnings: normal on CPU-only Kaggle.
- If you pass a TFRecord directly, use `--tfrecord /path/to/file.tfrecords` or positional path.
