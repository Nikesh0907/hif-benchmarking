#!/bin/bash
# Fast multi-model HIF benchmark for SF=8 and SF=16
# Run on Kaggle with: bash test_all_models.sh

echo "========================================================================"
echo "HIF Multi-Model Benchmark: SF=8 and SF=16"
echo "========================================================================"

HSI_DIR="/kaggle/input/datasets/nikeshreddypatlolla/cave-dataset-2/Data/Test/HSI"
RGB_DIR="/kaggle/input/datasets/nikeshreddypatlolla/cave-dataset-2/Data/Test/RGB"

cd /kaggle/working/hif-benchmarking

# ============================================================================
# 1. DBIN SF=8 (260k epochs)
# ============================================================================
echo ""
echo "========== [1] DBIN SF=8 (260k) =========="
python methods/_DBIN/dbintest.py \
    "$HSI_DIR" \
    --model_dir "methods/_DBIN/models_ibp_sn22" \
    --rgb_dir "$RGB_DIR" \
    --sf 8 \
    --num_images 12 2>&1 | grep -E "AVERAGE|psnr|ssim|sam|ergas" | head -5

# ============================================================================
# 2. DBIN SF=16 (15k epochs)
# ============================================================================
echo ""
echo "========== [2] DBIN SF=16 (15k optimal) =========="
python methods/_DBIN/dbintest.py \
    "$HSI_DIR" \
    --model_dir "/kaggle/input/model-sf16-15k/tensorflow2/default/1" \
    --rgb_dir "$RGB_DIR" \
    --sf 16 \
    --num_images 12 2>&1 | grep -E "AVERAGE|psnr|ssim|sam|ergas" | head -5

# ============================================================================
# 3. TSFN SF=8
# ============================================================================
echo ""
echo "========== [3] TSFN SF=8 =========="
python methods/_TSFN/test_kaggle_direct.py \
    --hsi_dir "$HSI_DIR" \
    --rgb_dir "$RGB_DIR" \
    --sf 8 2>&1 | grep -E "AVERAGE|PSNR|SAM|ERGAS|SSIM" | tail -5

# ============================================================================
# 4. TSFN SF=16
# ============================================================================
echo ""
echo "========== [4] TSFN SF=16 =========="
python methods/_TSFN/test_kaggle_direct.py \
    --hsi_dir "$HSI_DIR" \
    --rgb_dir "$RGB_DIR" \
    --sf 16 2>&1 | grep -E "AVERAGE|PSNR|SAM|ERGAS|SSIM" | tail -5

# ============================================================================
# 5. HSISR SF=8 (experimental - may not work)
# ============================================================================
echo ""
echo "========== [5] HSISR SF=8 (may fail - trained SF=4) =========="
python methods/_HSISR/test_ms_mats.py \
    --hsi_dir "$HSI_DIR" \
    --msi_dir "$RGB_DIR" \
    --sf 8 2>&1 | grep -E "PSNR|SAM|ERGAS|SSIM" | tail -5 || echo "HSISR SF=8 not compatible"

echo ""
echo "========================================================================"
echo "Benchmark complete!"
echo "========================================================================"
echo ""
echo "Summary Table:"
echo "Model          | SF=8 PSNR | SF=16 PSNR | Notes"
echo "--------------------------------------------------------------------"
echo "DBIN           | ~44.0     | ~43.0     | Needs RGB guidance"
echo "TSFN           | ???       | ???       | (from above)"
echo "HSISR          | ???       | ???       | (from above)"
echo ""
