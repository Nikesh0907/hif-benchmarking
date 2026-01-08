#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and evaluate CUCaNet on a multi-scene split.

CUCaNet's official pipeline expects data from its dataloader (keys: lhsi/hmsi/hhsi).
This script supports a clean 2-stage workflow:

1) Train mode: use ALL training scenes each epoch and save checkpoints.
2) Eval mode: load a saved checkpoint and evaluate on ALL test scenes.

It stages your .mat files into methods/_CUCaNet/CAVE/{train,test}/ and ensures the
required variable name is present: it converts 'hsi' -> 'img' when needed.
"""

import argparse
import os
import sys
import glob
import shutil
from pathlib import Path
import copy

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


def _to_hwc(arr_chw):
    if arr_chw.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W), got {arr_chw.shape}")
    return arr_chw.transpose(1, 2, 0)


def _compute_psnr(ref_hwc, tar_hwc, eps=1e-12):
    ref = ref_hwc.astype(np.float64)
    tar = tar_hwc.astype(np.float64)
    mse = np.mean((ref - tar) ** 2)
    max2 = float(np.max(ref) ** 2)
    return float(10.0 * np.log10((max2 + eps) / (mse + eps)))


def _compute_sam_deg(ref_hwc, tar_hwc, eps=1e-12):
    ref = ref_hwc.astype(np.float64)
    tar = tar_hwc.astype(np.float64)
    ref_flat = ref.reshape(-1, ref.shape[2])
    tar_flat = tar.reshape(-1, tar.shape[2])
    dot = np.sum(ref_flat * tar_flat, axis=1)
    ref_norm = np.linalg.norm(ref_flat, axis=1)
    tar_norm = np.linalg.norm(tar_flat, axis=1)
    cosang = dot / (ref_norm * tar_norm + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.mean(ang) * (180.0 / np.pi))


def _compute_ergas(ref_hwc, tar_hwc, sf=32, eps=1e-12):
    ref = ref_hwc.astype(np.float64)
    tar = tar_hwc.astype(np.float64)
    bands = ref.shape[2]
    ergas_acc = 0.0
    for b in range(bands):
        ref_b = ref[:, :, b]
        tar_b = tar[:, :, b]
        mse_b = np.mean((ref_b - tar_b) ** 2)
        mean_b = np.mean(tar_b)
        ergas_acc += mse_b / (mean_b ** 2 + eps)
    return float(100.0 / sf * np.sqrt(ergas_acc / bands))


def _compute_ssim(ref_hwc, tar_hwc):
    # Match DBIN-style: compute SSIM per band, average.
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        try:
            from skimage.measure import compare_ssim as ssim  # older skimage
        except Exception:
            return None

    ref = ref_hwc.astype(np.float32)
    tar = tar_hwc.astype(np.float32)
    bands = ref.shape[2]
    vals = []
    for b in range(bands):
        vals.append(ssim(ref[:, :, b], tar[:, :, b], data_range=1.0))
    return float(np.mean(vals))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _copy_and_fix_mats(src_paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    out_paths = []
    for src in src_paths:
        name = Path(src).stem
        dst = os.path.join(dst_dir, name + ".mat")
        print(f"Copying {src} to {dst}")
        shutil.copy(src, dst)

        mat_data = sio.loadmat(dst)
        if 'img' not in mat_data and 'hsi' in mat_data:
            mat_data['img'] = mat_data.pop('hsi')
            sio.savemat(dst, mat_data)
            print(f"Renamed 'hsi' key to 'img' in {dst}")
        out_paths.append(dst)
    return out_paths


def _parse_scene_name(name_field):
    # dataloader returns a list/tuple of length 1 for batchsize=1
    if isinstance(name_field, (list, tuple)):
        return ''.join([str(x) for x in name_field])
    return str(name_field)


def stage_split(train_mats, test_mats, cave_dir="./CAVE"):
    cave_dir = os.path.abspath(cave_dir)
    cave_train_dir = os.path.join(cave_dir, "train")
    cave_test_dir = os.path.join(cave_dir, "test")
    os.makedirs(cave_train_dir, exist_ok=True)
    os.makedirs(cave_test_dir, exist_ok=True)
    if train_mats is not None:
        _copy_and_fix_mats(train_mats, cave_train_dir)
    if test_mats is not None:
        _copy_and_fix_mats(test_mats, cave_test_dir)


def train_multi_scene(train_opt, cave_dir="./CAVE"):
    """Train on ALL scenes under CAVE/train/*.mat and save checkpoints."""
    cave_dir = os.path.abspath(cave_dir)
    checkpoint_dir = os.path.join(train_opt.checkpoints_dir, train_opt.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train dataloader over all train mats
    train_opt.mat_name = "train/*.mat"
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    print(f"Training scenes (used every epoch): {dataset_size}")

    train_model = create_model(train_opt,
                               train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)
    train_model.setup(train_opt)

    total_epochs = int(train_opt.niter + train_opt.niter_decay)
    print(f"Training for {total_epochs} epochs...")

    best_psnr = -1e9
    best_epoch = None

    for epoch in range(1, total_epochs + 1):
        epoch_psnrs = []
        for _, data in enumerate(train_dataloader):
            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)
            epoch_psnrs.append(train_model.cal_psnr())

        avg_psnr = float(np.mean(epoch_psnrs)) if epoch_psnrs else float('nan')
        if epoch % train_opt.print_freq == 0 or epoch == 1 or epoch == total_epochs:
            print(f"Epoch {epoch}/{total_epochs}, avg train PSNR: {avg_psnr:.4f}")

        # Save periodic checkpoints
        if train_opt.save_epoch_freq and (epoch % int(train_opt.save_epoch_freq) == 0):
            train_model.save_networks(f"epoch_{epoch}")

        # Save best checkpoint (based on training PSNR)
        if np.isfinite(avg_psnr) and avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            train_model.save_networks("best")
            with open(os.path.join(checkpoint_dir, "best.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_epoch={best_epoch}\n")
                f.write(f"best_avg_train_psnr={best_psnr}\n")

        train_model.update_learning_rate(avg_psnr)

    print(f"Training complete. Best avg train PSNR {best_psnr:.4f} at epoch {best_epoch}.")
    print(f"Checkpoints saved under: {checkpoint_dir}")


def eval_multi_scene(eval_opt, which_epoch, cave_dir="./CAVE"):
    """Load checkpoint and evaluate on ALL scenes under CAVE/test/*.mat."""
    cave_dir = os.path.abspath(cave_dir)
    results_dir = os.path.join(eval_opt.results_dir, "CAVE", eval_opt.name, str(which_epoch))
    os.makedirs(results_dir, exist_ok=True)

    eval_opt.isTrain = False
    eval_opt.batchsize = 1
    eval_opt.mat_name = "test/*.mat"
    test_dataloader = get_dataloader(eval_opt, isTrain=False)
    print(f"Testing scenes: {len(test_dataloader)}")

    eval_model = create_model(eval_opt,
                              test_dataloader.hsi_channels,
                              test_dataloader.msi_channels,
                              test_dataloader.lrhsi_height,
                              test_dataloader.lrhsi_width,
                              test_dataloader.sp_matrix,
                              test_dataloader.sp_range)
    eval_model.setup(eval_opt)
    eval_model.load_networks(str(which_epoch))
    eval_model.eval()

    metric_rows = []
    for _, data in enumerate(test_dataloader):
        scene_name = _parse_scene_name(data['name'])
        print(f"  Testing {scene_name}...", end=" ")

        eval_model.set_input(data, False)
        eval_model.test()

        visuals = eval_model.get_current_visuals()
        key_name = eval_model.get_visual_corresponding_name()['real_hhsi']
        rec_hsi = visuals[key_name].data.cpu().float().numpy()[0]  # (C,H,W)
        rec_hsi_hwc = _to_hwc(rec_hsi)

        real_hsi = data['hhsi'].cpu().float().numpy()[0]  # (C,H,W)
        real_hsi_hwc = _to_hwc(real_hsi)

        psnr = _compute_psnr(real_hsi_hwc, rec_hsi_hwc)
        sam = _compute_sam_deg(real_hsi_hwc, rec_hsi_hwc)
        ergas = _compute_ergas(real_hsi_hwc, rec_hsi_hwc, sf=int(eval_opt.scale_factor))
        ssim_val = _compute_ssim(real_hsi_hwc, rec_hsi_hwc)

        if ssim_val is None:
            print(f"PSNR {psnr:.3f} | SAM {sam:.3f} | ERGAS {ergas:.3f} | SSIM n/a")
        else:
            print(f"PSNR {psnr:.3f} | SAM {sam:.3f} | ERGAS {ergas:.3f} | SSIM {ssim_val:.4f}")

        out_path = os.path.join(results_dir, f"{scene_name}.mat")
        sio.savemat(out_path, {'out': rec_hsi_hwc})

        metric_rows.append((scene_name, psnr, sam, ergas, ssim_val))

    if metric_rows:
        avg_psnr = float(np.mean([r[1] for r in metric_rows]))
        avg_sam = float(np.mean([r[2] for r in metric_rows]))
        avg_ergas = float(np.mean([r[3] for r in metric_rows]))
        ssim_vals = [r[4] for r in metric_rows if r[4] is not None]
        avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else None
        if avg_ssim is None:
            print(f"\nAverages on test set: PSNR {avg_psnr:.3f} | SAM {avg_sam:.3f} | ERGAS {avg_ergas:.3f} | SSIM n/a")
        else:
            print(f"\nAverages on test set: PSNR {avg_psnr:.3f} | SAM {avg_sam:.3f} | ERGAS {avg_ergas:.3f} | SSIM {avg_ssim:.4f}")
        with open(os.path.join(results_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"avg_psnr={avg_psnr}\n")
            f.write(f"avg_sam={avg_sam}\n")
            f.write(f"avg_ergas={avg_ergas}\n")
            if avg_ssim is not None:
                f.write(f"avg_ssim={avg_ssim}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    ap.add_argument("--train_dir", default=None, help="Directory with training HSI .mat files (needed for train/train_eval unless already staged)")
    ap.add_argument("--test_dir", default=None, help="Directory with test HSI .mat files (optional; eval can use already-staged ./CAVE/test)")
    ap.add_argument("--srf_name", default="Nikon_D700_Qu", help="Spectral response function name")
    ap.add_argument("--scale_factor", type=int, default=32, help="Scale factor")
    ap.add_argument("--epochs", type=int, default=None, help="Total training epochs (sets niter=epochs, niter_decay=0)")
    ap.add_argument("--niter", type=int, default=2000, help="Number of training epochs at starting learning rate")
    ap.add_argument("--niter_decay", type=int, default=8000, help="Number of training epochs to linearly decay")
    ap.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    ap.add_argument("--batchsize", type=int, default=1, help="Batch size (keep 1 for CUCaNet)")
    ap.add_argument("--save_every", type=int, default=20, help="Save checkpoint every N epochs")
    ap.add_argument("--which_epoch", type=str, default="best", help="Checkpoint tag to load for eval: best or epoch_XX")
    ap.add_argument("--exp_name", type=str, default="CAVE_20train", help="Experiment folder name under checkpoints")
    ap.add_argument("--max_train", type=int, default=999, help="Max number of training files to use")
    ap.add_argument("--max_test", type=int, default=999, help="Max number of test files to use")
    args = ap.parse_args()

    # Resolve / validate inputs depending on mode
    train_mats = None
    test_mats = None
    if args.mode in ("train", "train_eval"):
        if not args.train_dir:
            raise SystemExit("--train_dir is required for mode=train/train_eval")
        train_mats = sorted(glob.glob(os.path.join(args.train_dir, "*.mat")))[:args.max_train]
        if not train_mats:
            raise SystemExit(f"No training .mat files found in {args.train_dir}")
        if args.test_dir:
            test_mats = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))[:args.max_test]
            if not test_mats:
                raise SystemExit(f"No test .mat files found in {args.test_dir}")
        print(f"Found {len(train_mats)} training files" + (f" and {len(test_mats)} test files" if test_mats else ""))
    elif args.mode == "eval":
        if args.test_dir:
            test_mats = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))[:args.max_test]
            if not test_mats:
                raise SystemExit(f"No test .mat files found in {args.test_dir}")
    
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
    if args.epochs is not None:
        train_opt.niter = int(args.epochs)
        train_opt.niter_decay = 0
    else:
        train_opt.niter = args.niter
        train_opt.niter_decay = args.niter_decay
    train_opt.lr = args.lr
    train_opt.batchsize = args.batchsize
    total_epochs = train_opt.niter + train_opt.niter_decay
    train_opt.print_freq = max(1, total_epochs // 10)
    train_opt.save_epoch_freq = int(args.save_every)
    train_opt.checkpoints_dir = "./checkpoints"
    train_opt.results_dir = "./Results"

    # Use a single experiment name for the whole split
    train_opt.name = args.exp_name

    # Stage mats only when sources are provided; eval-only can reuse existing staging.
    if train_mats is not None or test_mats is not None:
        print(f"\n{'='*80}")
        print("Staging mats under ./CAVE/{train,test}")
        print(f"{'='*80}")
        stage_split(train_mats, test_mats, cave_dir="./CAVE")

    # Ensure staged data exists for the selected mode
    if args.mode in ("train", "train_eval"):
        staged_train = sorted(glob.glob(os.path.join("./CAVE", "train", "*.mat")))
        if not staged_train:
            raise SystemExit("No staged training mats found under ./CAVE/train/*.mat")
    if args.mode in ("eval", "train_eval"):
        staged_test = sorted(glob.glob(os.path.join("./CAVE", "test", "*.mat")))
        if not staged_test:
            raise SystemExit("No staged test mats found under ./CAVE/test/*.mat")

    if args.mode in ("train", "train_eval"):
        train_multi_scene(train_opt, cave_dir="./CAVE")

    if args.mode in ("eval", "train_eval"):
        eval_opt = copy.deepcopy(train_opt)
        eval_multi_scene(eval_opt, which_epoch=args.which_epoch, cave_dir="./CAVE")
    
    print(f"\n{'='*80}")
    print("Done")
    print(f"Checkpoints: ./checkpoints/{train_opt.name}/")
    print(f"Results: ./Results/CAVE/{train_opt.name}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    setup_seed(1)
    main()
