#!/usr/bin/env python3
"""Pure-Python port of TSFN's MATLAB post-enhancement (enhance_adaptive.m).

This script refines TSFN's raw network output (HR-HSI) using the observation
models and an adaptive parameter search (golden section), solving a Sylvester
equation in the Fourier domain.

It is adapted from the MATLAB code shipped in this repo under:
  methods/_TSFN/enhance_adaptive.m
  methods/_TSFN/enhancement/*.m

Inputs:
- A directory containing per-image .mat files with keys:
    - 'sr': predicted HR HSI, shape [H,W,31]
    - 'gt': ground-truth HR HSI, shape [H,W,31]
  (This matches the outputs written by methods/_TSFN/save_image.py.)

Outputs:
- Writes enhanced results as .mat files with keys {'sr','gt'} to --out_dir.
- Prints before/after metrics using tools/hif_metrics.py.

Kaggle usage example:
  !pip -q install -r auxiliary/requirements.txt
  !python methods/_TSFN/enhancement/enhance_adaptive.py \
      --in_dir /kaggle/working/tsfn_out_test \
      --out_dir /kaggle/working/tsfn_out_test_enh \
      --sf 8 --kernel gaussian

Notes:
- The original MATLAB script uses sf=8 by default.
- Runtime can be noticeable because each image runs a golden-section search
  that repeatedly solves a Sylvester equation. Use --limit to test quickly.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import scipy.io

# Make imports work no matter what the current working directory is.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.hif_metrics import compute_metrics, normalize01


def _gaussian_psf(size: int = 8, sigma: float = 3.0) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum() if psf.sum() != 0 else 1.0
    return psf.astype(np.float64)


def _psf2otf(psf: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """MATLAB-like psf2otf for 2D PSFs."""
    oh, ow = out_hw
    psf = np.asarray(psf, dtype=np.float64)
    ph, pw = psf.shape

    pad = np.zeros((oh, ow), dtype=np.float64)
    pad[:ph, :pw] = psf

    # Circular shift so that PSF center is at (0,0)
    pad = np.roll(pad, -int(ph // 2), axis=0)
    pad = np.roll(pad, -int(pw // 2), axis=1)

    return np.fft.fft2(pad)


def hyperConvert2D(img3d: np.ndarray) -> np.ndarray:
    """MATLAB hyperConvert2D: [H,W,C] -> [C, H*W] using column-major order."""
    if img3d.ndim == 2:
        h, w = img3d.shape
        c = 1
        arr = img3d.reshape((h * w, 1), order="F")
        return arr.T
    h, w, c = img3d.shape
    arr = img3d.reshape((h * w, c), order="F")
    return arr.T


def hyperConvert3D(img2d: np.ndarray, h: int, w: int) -> np.ndarray:
    """MATLAB hyperConvert3D: [C, H*W] -> [H,W,C] using column-major order."""
    img2d = np.asarray(img2d)
    c, n = img2d.shape
    if n == 1:
        return img2d.reshape((h, w), order="F")
    arr = img2d.T.reshape((h, w, c), order="F")
    return arr


def create_F() -> np.ndarray:
    F = np.array(
        [
            [
                2,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                6,
                11,
                17,
                21,
                22,
                21,
                20,
                20,
                19,
                19,
                18,
                18,
                17,
                17,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                4,
                6,
                8,
                11,
                16,
                19,
                21,
                20,
                18,
                16,
                14,
                11,
                7,
                5,
                3,
                2,
                2,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
            ],
            [
                7,
                10,
                15,
                19,
                25,
                29,
                30,
                29,
                27,
                22,
                16,
                9,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ],
        dtype=np.float64,
    )

    for band in range(3):
        div = float(F[band, :].sum())
        if div == 0:
            continue
        F[band, :] = F[band, :] / div
    return F


@dataclass(frozen=True)
class Parameters:
    h: int
    w: int
    sf: int
    fft_B: np.ndarray
    fft_BT: np.ndarray

    def H(self, z: np.ndarray) -> np.ndarray:
        return H_z(z, self.fft_B, self.sf, (self.h, self.w))

    def HT(self, y: np.ndarray) -> np.ndarray:
        return HT_y(y, self.fft_BT, self.sf, (self.h, self.w))


def Parameters_setting(sf: int, kernel: str, sz: Tuple[int, int]) -> Parameters:
    if kernel.lower() in ("uniform", "uniform_blur"):
        psf = np.ones((sf, sf), dtype=np.float64) / float(sf * sf)
    elif kernel.lower() in ("gaussian", "gaussian_blur"):
        psf = _gaussian_psf(8, 3.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    fft_B = _psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return Parameters(h=sz[0], w=sz[1], sf=sf, fft_B=fft_B, fft_BT=fft_BT)


def H_z(z: np.ndarray, fft_B: np.ndarray, sf: int, sz: Tuple[int, int]) -> np.ndarray:
    """Blur + decimate operator in the spectral (per-band) unfolded domain."""
    z = np.asarray(z)
    ch, n = z.shape
    h, w = sz
    start = 0  # MATLAB s0=1, but 0-indexed

    if ch == 1:
        Hz = np.real(np.fft.ifft2(np.fft.fft2(z.reshape((h, w), order="F")) * fft_B))
        t = Hz[start::sf, start::sf]
        return t.reshape((1, -1), order="F")

    out = np.zeros((ch, (h // sf) * (w // sf)), dtype=np.float64)
    for i in range(ch):
        Hz = np.real(
            np.fft.ifft2(np.fft.fft2(z[i, :].reshape((h, w), order="F")) * fft_B)
        )
        t = Hz[start::sf, start::sf]
        out[i, :] = t.reshape((-1,), order="F")
    return out


def HT_y(y: np.ndarray, fft_BT: np.ndarray, sf: int, sz: Tuple[int, int]) -> np.ndarray:
    """Adjoint of H_z."""
    y = np.asarray(y)
    ch, n = y.shape
    h, w = sz
    start = 0

    if ch == 1:
        z = np.zeros((h, w), dtype=np.float64)
        z[start::sf, start::sf] = y.reshape((h // sf, w // sf), order="F")
        Htz = np.real(np.fft.ifft2(np.fft.fft2(z) * fft_BT))
        return Htz.reshape((1, -1), order="F")

    out = np.zeros((ch, h * w), dtype=np.float64)
    for i in range(ch):
        t = np.zeros((h, w), dtype=np.float64)
        t[start::sf, start::sf] = y[i, :].reshape((h // sf, w // sf), order="F")
        Htz = np.real(np.fft.ifft2(np.fft.fft2(t) * fft_BT))
        out[i, :] = Htz.reshape((-1,), order="F")
    return out


def pplus(X: np.ndarray, n_dr: int, n_dc: int) -> np.ndarray:
    """Sum all blocks into the first block (P^{-1} multiply)."""
    X = np.asarray(X)
    nr, nc, nb = X.shape
    dr = nr // n_dr
    dc = nc // n_dc

    sum_block = np.zeros((n_dr, n_dc, nb), dtype=X.dtype)
    for bi in range(dr):
        for bj in range(dc):
            sum_block += X[bi * n_dr : (bi + 1) * n_dr, bj * n_dc : (bj + 1) * n_dc, :]

    out = X.copy()
    out[:n_dr, :n_dc, :] = sum_block
    return out


def pplus_s(X: np.ndarray, n_dr: int, n_dc: int) -> np.ndarray:
    X = np.asarray(X)
    nr, nc, nb = X.shape
    dr = nr // n_dr
    dc = nc // n_dc

    sum_block = np.zeros((n_dr, n_dc, nb), dtype=X.dtype)
    for bi in range(dr):
        for bj in range(dc):
            sum_block += X[bi * n_dr : (bi + 1) * n_dr, bj * n_dc : (bj + 1) * n_dc, :]
    return sum_block


def sylvester(C1: np.ndarray, FBm: np.ndarray, ds_r: int, n_dr: int, n_dc: int, C3: np.ndarray) -> np.ndarray:
    """Port of Sylvester.m.

    Returns Z (L x (nr*nc)).
    """
    L = C1.shape[1]
    nr = ds_r * n_dr
    nc = ds_r * n_dc

    FBmC = np.conj(FBm)
    FBs = np.repeat(FBm[:, :, None], L, axis=2)
    FBCs1 = np.repeat(FBmC[:, :, None], L, axis=2)

    # Eigendecomposition
    eigvals, Q = np.linalg.eig(C1)
    # numpy returns eigenvectors as columns in Q
    Lambda = eigvals.reshape((1, 1, L)).astype(np.complex128)

    InvLbd = 1.0 / np.repeat(Lambda, nr, axis=0)
    InvLbd = np.repeat(InvLbd, nc, axis=1)

    B2Sum = pplus((np.abs(FBs) ** 2) / (ds_r**2), n_dr, n_dc)
    InvDI = 1.0 / (
        B2Sum[:n_dr, :n_dc, :] + np.repeat(Lambda, n_dr, axis=0).repeat(n_dc, axis=1)
    )

    # Q\C3 in MATLAB -> solve(Q, C3)
    QC3 = np.linalg.solve(Q, C3.astype(np.complex128))
    C30 = np.fft.fft2(QC3.T.reshape((nr, nc, L), order="F"), axes=(0, 1)) * InvLbd

    temp = pplus_s((C30 / (ds_r**2)) * FBs, n_dr, n_dc)

    invQUF = C30 - np.tile(temp * InvDI, (ds_r, ds_r, 1)) * FBCs1

    VXF = (Q @ invQUF.reshape((nr * nc, L), order="F").T).T  # (nr*nc, L)

    Z_cube = np.fft.ifft2(VXF.reshape((nr, nc, L), order="F"), axes=(0, 1))
    Z = np.real(Z_cube).reshape((nr * nc, L), order="F").T
    return Z


def calculate_J(
    mu: float,
    eta: float,
    par: Parameters,
    R: np.ndarray,
    X_CNN: np.ndarray,
    HSI3: np.ndarray,
    MSI3: np.ndarray,
    sf: int,
) -> Tuple[float, float, float, np.ndarray]:
    nr, nc, _ = X_CNN.shape

    HR_HSI3 = hyperConvert2D(X_CNN)

    H1 = eta * (R.T @ R) + mu * np.eye(R.shape[1], dtype=np.float64)
    HHH1 = par.HT(HSI3)
    H3 = eta * (R.T @ MSI3) + mu * HR_HSI3 + HHH1

    X_fin = sylvester(H1, par.fft_B, sf, nr // sf, nc // sf, H3)

    J1 = float(np.linalg.norm(HSI3 - par.H(X_fin), ord="fro") ** 2)
    J2 = float(np.linalg.norm(MSI3 - (R @ X_fin), ord="fro") ** 2)
    J3 = float(np.linalg.norm(X_fin - HR_HSI3, ord="fro") ** 2)

    return J1, J2, J3, X_fin


def mdc_dis(
    mu: float,
    eta: float,
    par: Parameters,
    R: np.ndarray,
    X_CNN: np.ndarray,
    HSI3: np.ndarray,
    MSI3: np.ndarray,
    sf: int,
    alpha: float,
    beta: float,
) -> float:
    J1, J2, J3, _ = calculate_J(mu, eta, par, R, X_CNN, HSI3, MSI3, sf)
    J3 = J3 * (alpha + beta)
    J12 = J1 + J2
    return float(np.sqrt((J12**2) + (J3**2)))


def search_2_gss(
    par: Parameters,
    R: np.ndarray,
    X_CNN: np.ndarray,
    HSI3: np.ndarray,
    MSI3: np.ndarray,
    sf: int,
    a: float = 1e-8,
    b: float = 1.0,
    ell: float = 0.001,
) -> Tuple[np.ndarray, float, float]:
    """Golden-section search with caching (faster than MATLAB script)."""

    alpha = (3.0 / 31.0) ** 2
    beta = (1.0 / (sf**2)) ** 2
    eta = 1.0

    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    f_c = mdc_dis(c, eta, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta)
    f_d = mdc_dis(d, eta, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta)

    while abs(c - d) > ell:
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - (b - a) / gr
            f_c = mdc_dis(c, eta, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta)
        else:
            a, c, f_c = c, d, f_d
            d = a + (b - a) / gr
            f_d = mdc_dis(d, eta, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta)

    mu_opti = ((b + a) / 2.0) * (alpha + beta)
    _, _, _, X_fin = calculate_J(mu_opti, eta, par, R, X_CNN, HSI3, MSI3, sf)
    return X_fin, float(mu_opti), float(eta)


def _load_result_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    m = scipy.io.loadmat(path)
    if "sr" not in m or "gt" not in m:
        raise KeyError(f"Expected keys 'sr' and 'gt' in {path}. Keys={list(m.keys())}")
    sr = np.asarray(m["sr"], dtype=np.float64)
    gt = np.asarray(m["gt"], dtype=np.float64)
    return sr, gt


def main() -> int:
    ap = argparse.ArgumentParser(description="TSFN MATLAB enhancement port (pure Python)")
    ap.add_argument("--in_dir", required=True, help="Directory with TSFN result mats (keys: sr, gt)")
    ap.add_argument("--out_dir", required=True, help="Output directory for enhanced mats")
    ap.add_argument("--sf", type=int, default=8, help="Scale factor (MATLAB default: 8)")
    ap.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "uniform"], help="Blur kernel")
    ap.add_argument("--limit", type=int, default=0)

    # Search params
    ap.add_argument("--mu_a", type=float, default=1e-8)
    ap.add_argument("--mu_b", type=float, default=1.0)
    ap.add_argument("--ell", type=float, default=0.001)
    ap.add_argument(
        "--no_search",
        action="store_true",
        help="Skip golden-section search and use --mu_fixed instead (faster, less faithful).",
    )
    ap.add_argument("--mu_fixed", type=float, default=1e-4)

    args = ap.parse_args()

    mats = sorted(glob.glob(os.path.join(args.in_dir, "*.mat")))
    if not mats:
        raise SystemExit(f"No .mat files found in: {args.in_dir}")

    if args.limit and args.limit > 0:
        mats = mats[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    R = create_F()  # 3x31

    sums_before: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "sam": 0.0, "ergas": 0.0}
    sums_after: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "sam": 0.0, "ergas": 0.0}

    for idx, path in enumerate(mats, start=1):
        name = Path(path).stem
        sr, gt = _load_result_mat(path)

        # Normalize and ensure HWC
        X_CNN = normalize01(sr)
        S = normalize01(gt)

        if X_CNN.ndim != 3 or S.ndim != 3:
            raise SystemExit(f"Unexpected shapes in {path}: sr={X_CNN.shape} gt={S.shape}")

        nr, nc, L = S.shape
        if L != 31:
            raise SystemExit(f"Expected 31-band HSI. Got {L} bands in {path}")
        if nr % args.sf != 0 or nc % args.sf != 0:
            raise SystemExit(f"Image size {nr}x{nc} not divisible by sf={args.sf} in {path}")

        # Unfold GT
        S_bar = hyperConvert2D(S)

        # Set operators
        par = Parameters_setting(args.sf, args.kernel, (nr, nc))

        # Simulate LR HSI and HR MSI in unfolded forms
        HSI3 = par.H(S_bar)
        MSI3 = R @ S_bar

        # Enhance
        if args.no_search:
            mu = float(args.mu_fixed)
            eta = 1.0
            _, _, _, X_fin = calculate_J(mu, eta, par, R, X_CNN, HSI3, MSI3, args.sf)
            mu_opti = mu
        else:
            X_fin, mu_opti, _ = search_2_gss(
                par,
                R,
                X_CNN,
                HSI3,
                MSI3,
                args.sf,
                a=float(args.mu_a),
                b=float(args.mu_b),
                ell=float(args.ell),
            )

        X_fin3 = hyperConvert3D(X_fin, nr, nc)
        X_fin3 = normalize01(X_fin3)

        mb = compute_metrics(S, X_CNN, ratio=args.sf)
        ma = compute_metrics(S, X_fin3, ratio=args.sf)
        for k in sums_before:
            sums_before[k] += mb[k]
            sums_after[k] += ma[k]

        print(
            f"[{idx}/{len(mats)}] {name} mu={mu_opti:.4e} "
            f"BEFORE PSNR={mb['psnr']:.3f} SSIM={mb['ssim']:.4f} SAM={mb['sam']:.4f} ERGAS={mb['ergas']:.3f} | "
            f"AFTER PSNR={ma['psnr']:.3f} SSIM={ma['ssim']:.4f} SAM={ma['sam']:.4f} ERGAS={ma['ergas']:.3f}"
        )

        scipy.io.savemat(os.path.join(args.out_dir, name + ".mat"), {"sr": X_fin3, "gt": S})

    n = float(len(mats))
    avg_b = {k: v / n for k, v in sums_before.items()}
    avg_a = {k: v / n for k, v in sums_after.items()}

    print(
        f"\nAVERAGE BEFORE (sf={args.sf}, n={len(mats)}): "
        f"PSNR={avg_b['psnr']:.3f} SSIM={avg_b['ssim']:.4f} SAM={avg_b['sam']:.4f} ERGAS={avg_b['ergas']:.3f}"
    )
    print(
        f"AVERAGE AFTER  (sf={args.sf}, n={len(mats)}): "
        f"PSNR={avg_a['psnr']:.3f} SSIM={avg_a['ssim']:.4f} SAM={avg_a['sam']:.4f} ERGAS={avg_a['ergas']:.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
