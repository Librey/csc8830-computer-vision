"""
===============================================================================
CSc 8830 Computer Vision, Module 4, Question 3
Frequency-domain edge detection and texture segmentation (NumPy FFT)
Author: Liberty Ikpeogu
===============================================================================

README
------
Install:
    pip install numpy opencv-python

Use from Python:
    import cv2, fourier_lab as fl
    f = cv2.imread("image.png", 0)
    out = fl.filter_image(f, "LoG band-pass", sigma=2.0)
    out["result"], out["H"], out["spectrum"], out["zero_crossings"]

    demo = fl.texture_demo(noise=0.6, sigma=6)
    demo["accuracy"]

Every filter runs the same way: pad the image, take the FFT, multiply by the
transfer function H(u, v), take the inverse FFT, crop. Frequencies are in
cycles per pixel, so the Nyquist limit is 0.5. The derivations are in
q3_fourier_edges_segmentation.pdf.
===============================================================================
"""
import numpy as np

FILTERS = [
    "Ideal high-pass",
    "Butterworth high-pass",
    "Gaussian high-pass",
    "Gradient magnitude (Gaussian-smoothed)",
    "LoG band-pass",
    "Gaussian low-pass",
]


def freq_axes(P, Q):
    """Frequency coordinates of an unshifted P x Q FFT, in cycles/pixel.
    fy runs down the rows, fx across the columns."""
    fy = np.fft.fftfreq(P)[:, None]
    fx = np.fft.fftfreq(Q)[None, :]
    return fx, fy


def transfer_function(name, P, Q, D0=0.05, order=2, sigma=2.0):
    """Return H(u, v) for the named filter on a P x Q frequency grid.
    For the gradient, returns the pair (Hx, Hy)."""
    fx, fy = freq_axes(P, Q)
    D = np.sqrt(fx ** 2 + fy ** 2)
    G = np.exp(-2 * np.pi ** 2 * sigma ** 2 * D ** 2)       # Gaussian low-pass, width sigma
    if name == "Ideal high-pass":
        return (D > D0).astype(float)
    if name == "Butterworth high-pass":
        with np.errstate(divide="ignore"):
            return 1.0 / (1.0 + (D0 / np.maximum(D, 1e-12)) ** (2 * order))
    if name == "Gaussian high-pass":
        return 1.0 - np.exp(-D ** 2 / (2 * D0 ** 2))
    if name == "Gradient magnitude (Gaussian-smoothed)":
        return (1j * 2 * np.pi * fx * G, 1j * 2 * np.pi * fy * G)
    if name == "LoG band-pass":
        return -4 * np.pi ** 2 * D ** 2 * G
    if name == "Gaussian low-pass":
        return np.exp(-D ** 2 / (2 * D0 ** 2))
    raise ValueError(name)


def zero_crossings(g, slope_k=1.0):
    """Pixels where g changes sign between neighbours with a jump bigger than
    slope_k standard deviations of g (weak crossings are noise)."""
    thr = slope_k * np.std(g)
    zc = np.zeros(g.shape, bool)
    a, b = g[:, :-1], g[:, 1:]
    zc[:, :-1] |= (np.sign(a) != np.sign(b)) & (np.abs(a - b) > thr)
    a, b = g[:-1, :], g[1:, :]
    zc[:-1, :] |= (np.sign(a) != np.sign(b)) & (np.abs(a - b) > thr)
    return zc


def filter_image(f, name, D0=0.05, order=2, sigma=2.0, pad=True):
    """Apply a frequency-domain filter to a grayscale image.

    Returns a dict with:
        result          filtered image (float; magnitude for the gradient)
        H               |H| shifted so zero frequency is in the centre
        spectrum        log(1 + |F|) of the input, shifted
        zero_crossings  boolean edge map (LoG only, else None)
    """
    f = f.astype(float) / 255.0
    M, N = f.shape
    # Zero-padding to 2M x 2N stops the periodic DFT from wrapping one edge
    # of the image onto the other. Pad with the mean to avoid a false border.
    P, Q = (2 * M, 2 * N) if pad else (M, N)
    fp = np.full((P, Q), f.mean())
    fp[:M, :N] = f
    F = np.fft.fft2(fp)
    H = transfer_function(name, P, Q, D0, order, sigma)

    if isinstance(H, tuple):
        gx = np.real(np.fft.ifft2(F * H[0]))[:M, :N]
        gy = np.real(np.fft.ifft2(F * H[1]))[:M, :N]
        result = np.hypot(gx, gy)
        Hmag = np.hypot(np.abs(H[0]), np.abs(H[1]))
    else:
        result = np.real(np.fft.ifft2(F * H))[:M, :N]
        Hmag = np.abs(H)

    Fs = np.fft.fftshift(np.fft.fft2(f))
    spectrum = np.log1p(np.abs(Fs))
    zc = zero_crossings(result) if name == "LoG band-pass" else None
    return dict(result=result, H=np.fft.fftshift(Hmag), spectrum=spectrum, zero_crossings=zc)


def radial_profile(name, D0=0.05, order=2, sigma=2.0, n=256):
    """|H| along the u axis from 0 to 0.5 cycles/pixel, for plotting."""
    H = transfer_function(name, 2, 2 * n, D0, order, sigma)
    if isinstance(H, tuple):
        H = np.hypot(np.abs(H[0]), np.abs(H[1]))
    row = np.abs(H[0, :n + 1])
    freqs = np.fft.fftfreq(2 * n)[:n + 1]
    freqs[-1] = 0.5
    return freqs, row


def texture_demo(noise=0.6, sigma=6.0, size=256, seed=1):
    """Two textures with equal mean brightness, separated by Gabor energy.

    Texture A: vertical stripes at 0.10 cycles/px.
    Texture B (ellipse): stripes at 0.06 cycles/px, rotated 60 degrees.
    Each Gabor filter is built in the frequency domain as a pair of Gaussians
    at +-(u0, v0); local energy is |filtered|^2 smoothed by a Gaussian; the
    boundary is the zero level set of log(E_B) - log(E_A).
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    tA = np.sin(2 * np.pi * 0.10 * x)
    ang = np.pi / 3
    tB = np.sin(2 * np.pi * 0.06 * (x * np.cos(ang) + y * np.sin(ang)))
    truth = ((x - 150) ** 2 / 80 ** 2 + (y - 120) ** 2 / 60 ** 2) < 1
    img = np.where(truth, tB, tA) + noise * rng.standard_normal((size, size))

    fx, fy = freq_axes(size, size)

    def gabor_pair(u0, v0):
        g = lambda a, b: np.exp(-2 * np.pi ** 2 * sigma ** 2 * ((fx - a) ** 2 + (fy - b) ** 2))
        return g(u0, v0) + g(-u0, -v0)

    HA = gabor_pair(0.10, 0.0)
    HB = gabor_pair(0.06 * np.cos(ang), 0.06 * np.sin(ang))
    F = np.fft.fft2(img)
    smooth = np.exp(-2 * np.pi ** 2 * 8 ** 2 * (fx ** 2 + fy ** 2))
    EA = np.real(np.fft.ifft2(np.fft.fft2(np.abs(np.fft.ifft2(F * HA)) ** 2) * smooth))
    EB = np.real(np.fft.ifft2(np.fft.fft2(np.abs(np.fft.ifft2(F * HB)) ** 2) * smooth))
    log_ratio = np.log(np.maximum(EB, 1e-9)) - np.log(np.maximum(EA, 1e-9))
    seg = log_ratio > 0
    return dict(image=img, spectrum=np.log1p(np.abs(np.fft.fftshift(F))),
                filters=np.fft.fftshift(HA + HB), log_ratio=log_ratio,
                segmentation=seg, truth=truth, accuracy=float((seg == truth).mean()))
