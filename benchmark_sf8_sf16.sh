#!/bin/bash
# Fast benchmark script for SF=8 and SF=16 across available models
# Usage: bash benchmark_sf8_sf16.sh

set -e

echo "========================================================================"
echo "HIF BENCHMARK: SF=8 and SF=16 Comparison"
echo "========================================================================"

HSI_DIR="/kaggle/input/cave-dataset-2/Data/Test/HSI"
RGB_DIR="/kaggle/input/cave-dataset-2/Data/Test/RGB"
NUM_IMAGES=12

cd /kaggle/working/hif-benchmarking

# ============================================================================
# 1. DBIN - TensorFlow, SF=8 and SF=16 (with RGB guidance)
# ============================================================================
echo ""
echo "[1/4] DBIN SF=8 (260k epochs)"
echo "----------------------------------------------------------------------"
python methods/_DBIN/dbintest.py \
    "$HSI_DIR" \
    --model_dir "methods/_DBIN/models_ibp_sn22" \
    --rgb_dir "$RGB_DIR" \
    --sf 8 \
    --num_images $NUM_IMAGES 2>&1 | grep -E "image|AVERAGE"

echo ""
echo "[1/4] DBIN SF=16 (15k epochs - known optimal)"
echo "----------------------------------------------------------------------"
python methods/_DBIN/dbintest.py \
    "$HSI_DIR" \
    --model_dir "/kaggle/input/model-sf16-15k/tensorflow2/default/1" \
    --rgb_dir "$RGB_DIR" \
    --sf 16 \
    --num_images $NUM_IMAGES 2>&1 | grep -E "image|AVERAGE"

# ============================================================================
# 2. TSFN - PyTorch, SF=8 and SF=16 (supports via degradation_mode)
# ============================================================================
echo ""
echo "[2/4] TSFN SF=8 (degradation_mode=0)"
echo "----------------------------------------------------------------------"
python methods/_TSFN/test_kaggle_direct.py \
    --hsi_dir "$HSI_DIR" \
    --rgb_dir "$RGB_DIR" \
    --sf 8 2>&1 | tail -20

echo ""
echo "[2/4] TSFN SF=16 (degradation_mode=2)"
echo "----------------------------------------------------------------------"
python methods/_TSFN/test_kaggle_direct.py \
    --hsi_dir "$HSI_DIR" \
    --rgb_dir "$RGB_DIR" \
    --sf 16 2>&1 | tail -20

# ============================================================================
# 3. HSISR - PyTorch, variable scale (test SF=8)
# ============================================================================
echo ""
echo "[3/4] HSISR SF=8"
echo "----------------------------------------------------------------------"
python methods/_HSISR/test_ms_mats.py \
    --hsi_dir "$HSI_DIR" \
    --msi_dir "$RGB_DIR" \
    --sf 8 2>&1 | tail -20 || echo "HSISR test failed or not available"

# ============================================================================
# 4. HSRnet - TensorFlow, SF=4 only (FIXED - can't test SF=8/16)
# ============================================================================
echo ""
echo "[4/4] HSRnet - WARNING: Only supports SF=4 (hardcoded)"
echo "----------------------------------------------------------------------"
echo "Skipping HSRnet: Not compatible with SF=8/16"

echo ""
echo "========================================================================"
echo "Benchmark complete!"
echo "========================================================================"
