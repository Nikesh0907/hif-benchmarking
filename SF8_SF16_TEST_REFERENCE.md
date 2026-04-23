# HIF Methods - Quick Test Reference (SF=8 & SF=16)

## Models with Pre-trained Weights Available

| Model | Weights Location | SF=8 | SF=16 | SF=32 | Status | Notes |
|-------|------------------|------|-------|-------|--------|-------|
| **DBIN** | methods/_DBIN/models_ibp_sn22/ | ✅ | ✅* | ❌ | Ready | (*15k optimal for SF=16, 260k for SF=8) |
| **TSFN** | methods/_TSFN/models/ssfsr_9layers_epoch500.pkl | ✅ | ✅ | ✅ | Ready | Supports all via degradation_mode |
| **HSISR** | methods/_HSISR/models/Cave_*.pth | ✅ | ⚠️ | ❌ | Ready | Only trained for SF=4, may not work SF=8/16 |
| **HSRnet** | methods/_HSRnet/models(cave)/ | ❌ | ❌ | ❌ | Skip | SF=4 hardcoded, cannot test SF=8/16 |
| **UDALN** | Need external | ❌ | ✅* | ❌ | Blocked | (*SF=5 trained, not SF=8) |
| **u2MDN** | Need external | ⚠️ | ⚠️ | ⚠️ | Blocked | No pre-trained (unsupervised) |
| **CMHFnet** | methods/_MHFnet/ | ❌ | ❌ | ❌ | Skip | RGB spectral mismatch issues |

---

## Quick Test Commands

### DBIN (Best for SF=16 comparison)
```bash
# SF=8 (260k epochs)
python methods/_DBIN/dbintest.py /kaggle/input/cave-dataset-2/Data/Test/HSI \
    --model_dir methods/_DBIN/models_ibp_sn22 \
    --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
    --sf 8 --num_images 12

# SF=16 (15k epochs - optimal)
python methods/_DBIN/dbintest.py /kaggle/input/cave-dataset-2/Data/Test/HSI \
    --model_dir /kaggle/input/model-sf16-15k/tensorflow2/default/1 \
    --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
    --sf 16 --num_images 12
```

### TSFN (Can test both SF=8 and SF=16)
```bash
# SF=8
python methods/_TSFN/test_kaggle_direct.py \
    --hsi_dir /kaggle/input/cave-dataset-2/Data/Test/HSI \
    --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
    --sf 8

# SF=16
python methods/_TSFN/test_kaggle_direct.py \
    --hsi_dir /kaggle/input/cave-dataset-2/Data/Test/HSI \
    --rgb_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
    --sf 16
```

### HSISR (Experimental - may not work for SF=8/16)
```bash
python methods/_HSISR/test_ms_mats.py \
    --hsi_dir /kaggle/input/cave-dataset-2/Data/Test/HSI \
    --msi_dir /kaggle/input/cave-dataset-2/Data/Test/RGB \
    --sf 8
```

---

## Recommendation

**For SF=8 vs SF=16 comparison, test:**
1. ✅ **DBIN** - Most reliable, SF-specific weights, uses RGB guidance properly
2. ✅ **TSFN** - Can test both scales with same weights, interesting comparison

**Skip for now:**
- ❌ HSRnet (SF=4 only)
- ❌ HSISR (only trained SF=4)
- ❌ Others (no compatible weights)
