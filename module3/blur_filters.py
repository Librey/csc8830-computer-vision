#!/usr/bin/env python3
"""
blur_filters.py
===============
CSc 8830: Computer Vision -- Module 3 Assignment
Image blurring with linear filters, and verification that

    spatial convolution  ==  multiplication in the Fourier domain

--------------------------------------------------------------------------
README (how to run this script)
--------------------------------------------------------------------------
1. Install dependencies (Python 3.9+):

       pip install -r requirements.txt

2. Blur an image and validate spatial vs. Fourier results from the CLI:

       python blur_filters.py --input sample/cameraman.png \
                              --kernel gaussian --size 15 --sigma 2.5 \
                              --out results/

   Options:
       --input   path to an image (grayscale or RGB)
       --kernel  box | gaussian                     (default: gaussian)
       --size    odd kernel side length K           (default: 15)
       --sigma   Gaussian std-dev in pixels         (default: K/6)
       --out     folder for the output images       (default: results/)

   The script prints the error between the two methods and saves:
       original.png, blur_spatial.png, blur_fourier.png,
       difference_x1e12.png (difference amplified so it is visible),
       spectra.png (log-magnitude spectra of image, kernel and product)

3. Run the web application (all demos are accessible from the web page):

       python app.py        ->  open http://127.0.0.1:5000

--------------------------------------------------------------------------
What is implemented
--------------------------------------------------------------------------
* box_kernel / gaussian_kernel : normalised (sum = 1) blur kernels
* conv2d_spatial_full          : 2-D convolution written directly from the
                                 definition  g[i,j] = sum_m sum_n f[m,n] h[i-m, j-n]
                                 (no library convolution is used)
* conv2d_separable_full        : same result for the Gaussian using two 1-D passes
* conv2d_fft_full              : IFFT( FFT(f) * FFT(h) ) with zero padding to at least
                                 (H+K-1, W+K-1) so the DFT's circular convolution
                                 equals ordinary (linear) convolution
* conv2d_fft_circular_same     : the *un-padded* FFT version, kept only to show
                                 what goes wrong without padding (wrap-around)
* validate                     : runs both methods and reports the difference
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy.fft import next_fast_len  # picks an FFT-friendly size >= n (only affects speed)
from scipy.signal import convolve2d  # used ONLY as an independent reference check


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
def box_kernel(k: int) -> np.ndarray:
    """k x k box (averaging) filter. Weights sum to 1 so brightness is preserved."""
    _check_odd(k)
    return np.full((k, k), 1.0 / (k * k))


def gaussian_kernel_1d(k: int, sigma: float | None = None) -> np.ndarray:
    """1-D Gaussian of length k, normalised to sum to 1."""
    _check_odd(k)
    sigma = k / 6.0 if sigma is None else sigma  # default: kernel spans about +/-3 sigma
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel(k: int, sigma: float | None = None) -> np.ndarray:
    """k x k Gaussian. It is the outer product of two 1-D Gaussians (separable)."""
    g = gaussian_kernel_1d(k, sigma)
    return np.outer(g, g)


def make_kernel(kind: str, k: int, sigma: float | None = None) -> np.ndarray:
    if kind == "box":
        return box_kernel(k)
    if kind == "gaussian":
        return gaussian_kernel(k, sigma)
    raise ValueError(f"unknown kernel '{kind}' (use 'box' or 'gaussian')")


def _check_odd(k: int) -> None:
    if k < 1 or k % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")


# --------------------------------------------------------------------------
# Spatial-domain convolution (from the definition)
# --------------------------------------------------------------------------
def conv2d_spatial_full(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Full linear 2-D convolution, straight from the definition

        g[i, j] = sum_a sum_b h[a, b] * f[i - a, j - b]

    Each kernel weight h[a, b] contributes a copy of the whole image shifted by
    (a, b). Summing these shifted copies is exactly the flip-and-slide sum, and
    the flip of the kernel is built into the index i - a, j - b.
    Output size is (H + kh - 1) x (W + kw - 1); pixels outside the image are 0.
    """
    H, W = f.shape
    kh, kw = h.shape
    out = np.zeros((H + kh - 1, W + kw - 1), dtype=np.float64)
    for a in range(kh):
        for b in range(kw):
            out[a:a + H, b:b + W] += h[a, b] * f
    return out


def conv1d_full_along(f: np.ndarray, h1: np.ndarray, axis: int) -> np.ndarray:
    """Full 1-D convolution of every row (axis=1) or column (axis=0) with h1."""
    k = h1.size
    if axis == 1:
        H, W = f.shape
        out = np.zeros((H, W + k - 1))
        for a in range(k):
            out[:, a:a + W] += h1[a] * f
    else:
        H, W = f.shape
        out = np.zeros((H + k - 1, W))
        for a in range(k):
            out[a:a + H, :] += h1[a] * f
    return out


def conv2d_separable_full(f: np.ndarray, g1d: np.ndarray) -> np.ndarray:
    """Convolution with outer(g1d, g1d) using two 1-D passes: 2K instead of K^2 multiplies."""
    return conv1d_full_along(conv1d_full_along(f, g1d, axis=1), g1d, axis=0)


def crop_same(full: np.ndarray, kh: int, kw: int, H: int, W: int) -> np.ndarray:
    """Crop the full convolution back to the input size (kernel centre at each pixel)."""
    r, c = kh // 2, kw // 2
    return full[r:r + H, c:c + W]


# --------------------------------------------------------------------------
# Fourier-domain convolution
# --------------------------------------------------------------------------
def fft_shape(H: int, W: int, kh: int, kw: int) -> tuple[int, int]:
    """
    Padded FFT size. It must be >= (H + kh - 1, W + kw - 1) so that circular
    convolution equals linear convolution. We round up to the next size with
    small prime factors because the FFT is much faster there (a size such as
    514 = 2 * 257 is slow); any size >= the minimum gives the same result.
    """
    return next_fast_len(H + kh - 1), next_fast_len(W + kw - 1)


def conv2d_fft_full(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Convolution theorem:  f * h  =  IFFT( FFT(f) . FFT(h) ).

    The DFT assumes both signals are periodic, so the product corresponds to
    *circular* convolution. If both arrays are zero-padded to at least
    (H + kh - 1) x (W + kw - 1) the periodic copies never overlap, and circular
    convolution becomes identical to linear (spatial) convolution.
    Returns the full (H + kh - 1) x (W + kw - 1) result.
    """
    H, W = f.shape
    kh, kw = h.shape
    shape = fft_shape(H, W, kh, kw)
    F = np.fft.fft2(f, s=shape)   # s= zero-pads at the bottom/right
    Hk = np.fft.fft2(h, s=shape)
    return np.real(np.fft.ifft2(F * Hk))[:H + kh - 1, :W + kw - 1]


def conv2d_fft_circular_same(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    FFT filtering WITHOUT padding: kernel is placed in an H x W array with its
    centre at the origin. Result is circular convolution, so pixels near one
    border are blurred with pixels from the opposite border (wrap-around).
    Included only to demonstrate why padding is required.
    """
    H, W = f.shape
    kh, kw = h.shape
    hp = np.zeros((H, W))
    hp[:kh, :kw] = h
    hp = np.roll(hp, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(f) * np.fft.fft2(hp)))


# --------------------------------------------------------------------------
# Helpers for images / spectra
# --------------------------------------------------------------------------
def load_image(path: str, max_side: int | None = None) -> np.ndarray:
    """Load image as float64 array, shape (H, W) or (H, W, 3), values 0..255."""
    im = Image.open(path)
    im = im.convert("L") if im.mode in ("L", "LA", "1", "I", "I;16", "F") else im.convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def to_uint8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def apply_per_channel(img: np.ndarray, fn) -> np.ndarray:
    """Apply a 2-D function to a gray image or to each channel of an RGB image."""
    if img.ndim == 2:
        return fn(img)
    return np.dstack([fn(img[..., c]) for c in range(img.shape[2])])


def log_spectrum(a: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """log(1 + |FFT|), zero frequency shifted to the centre (for display)."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a, s=shape))))


# --------------------------------------------------------------------------
# Validation: spatial vs. Fourier
# --------------------------------------------------------------------------
def validate(img: np.ndarray, kernel: np.ndarray, zero_pad: bool = True) -> dict:
    """
    Blur `img` with `kernel` using (1) the spatial definition, (2) the FFT, and
    (3) scipy's convolve2d as an independent reference. Returns images + metrics.

    zero_pad=False swaps in the un-padded (circular) FFT to show wrap-around error.
    """
    kh, kw = kernel.shape
    H, W = img.shape[:2]

    t0 = time.perf_counter()
    spatial_full = apply_per_channel(img, lambda ch: conv2d_spatial_full(ch, kernel))
    t_spatial = time.perf_counter() - t0
    spatial = apply_per_channel(spatial_full, lambda ch: crop_same(ch, kh, kw, H, W))

    t0 = time.perf_counter()
    if zero_pad:
        fourier_full = apply_per_channel(img, lambda ch: conv2d_fft_full(ch, kernel))
        fourier = apply_per_channel(fourier_full, lambda ch: crop_same(ch, kh, kw, H, W))
    else:
        fourier = apply_per_channel(img, lambda ch: conv2d_fft_circular_same(ch, kernel))
    t_fft = time.perf_counter() - t0

    reference = apply_per_channel(
        img, lambda ch: convolve2d(ch, kernel, mode="same", boundary="fill", fillvalue=0)
    )

    diff = spatial - fourier
    return {
        "spatial": spatial,
        "fourier": fourier,
        "reference": reference,
        "diff": diff,
        "metrics": {
            "max_abs_error": float(np.max(np.abs(diff))),
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "spatial_vs_scipy_max_abs": float(np.max(np.abs(spatial - reference))),
            "fourier_vs_scipy_max_abs": float(np.max(np.abs(fourier - reference))),
            "kernel_sum": float(kernel.sum()),
            "mean_in": float(img.mean()),
            "mean_spatial": float(spatial.mean()),
            "t_spatial_s": t_spatial,
            "t_fft_s": t_fft,
        },
    }


def benchmark(img2d: np.ndarray, sizes, sigma_fn=None, repeats: int = 1) -> list[dict]:
    """Time direct, separable and FFT Gaussian filtering for several kernel sizes."""
    rows = []
    for k in sizes:
        g1 = gaussian_kernel_1d(k, None if sigma_fn is None else sigma_fn(k))
        h = np.outer(g1, g1)
        best = {"direct": 1e9, "separable": 1e9, "fft": 1e9}
        for _ in range(repeats):
            t = time.perf_counter(); conv2d_spatial_full(img2d, h); best["direct"] = min(best["direct"], time.perf_counter() - t)
            t = time.perf_counter(); conv2d_separable_full(img2d, g1); best["separable"] = min(best["separable"], time.perf_counter() - t)
            t = time.perf_counter(); conv2d_fft_full(img2d, h); best["fft"] = min(best["fft"], time.perf_counter() - t)
        rows.append({"k": k, **best})
    return rows


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------
def _save_spectra_figure(img2d: np.ndarray, kernel: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = img2d.shape
    kh, kw = kernel.shape
    shape = fft_shape(H, W, kh, kw)
    S_img = log_spectrum(img2d, shape)
    S_ker = log_spectrum(kernel, shape)
    prod = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img2d, s=shape) * np.fft.fft2(kernel, s=shape))))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, s, t in zip(ax, (S_img, S_ker, prod),
                       ("log|F(u,v)|  image", "|H(u,v)|  kernel (low-pass)", "log|F(u,v)H(u,v)|  product")):
        a.imshow(s, cmap="magma"); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Blur an image spatially and in the Fourier domain and compare.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--kernel", choices=["box", "gaussian"], default="gaussian")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = load_image(args.input)
    kernel = make_kernel(args.kernel, args.size, args.sigma)
    res = validate(img, kernel)

    Image.fromarray(to_uint8(img)).save(os.path.join(args.out, "original.png"))
    Image.fromarray(to_uint8(res["spatial"])).save(os.path.join(args.out, "blur_spatial.png"))
    Image.fromarray(to_uint8(res["fourier"])).save(os.path.join(args.out, "blur_fourier.png"))
    Image.fromarray(to_uint8(np.abs(res["diff"]) * 1e12)).save(os.path.join(args.out, "difference_x1e12.png"))
    gray = img if img.ndim == 2 else img.mean(axis=2)
    _save_spectra_figure(gray, kernel, os.path.join(args.out, "spectra.png"))

    m = res["metrics"]
    print(f"kernel: {args.kernel} {args.size}x{args.size}   (sum of weights = {m['kernel_sum']:.6f})")
    print(f"max |spatial - fourier|        : {m['max_abs_error']:.3e}")
    print(f"mean |spatial - fourier|       : {m['mean_abs_error']:.3e}")
    print(f"RMSE                           : {m['rmse']:.3e}")
    print(f"max |spatial - scipy|          : {m['spatial_vs_scipy_max_abs']:.3e}")
    print(f"max |fourier - scipy|          : {m['fourier_vs_scipy_max_abs']:.3e}")
    print(f"time spatial / fft             : {m['t_spatial_s']*1e3:.1f} ms / {m['t_fft_s']*1e3:.1f} ms")
    print(f"images written to {args.out}/")


if __name__ == "__main__":
    main()
