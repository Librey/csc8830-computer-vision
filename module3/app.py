#!/usr/bin/env python3
"""
app.py -- Web application for CSc 8830 Module 3
================================================
README
------
    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000

The page lets you upload an image (or use a bundled sample), choose a box or
Gaussian kernel, and see:
  * the image blurred in the SPATIAL domain (direct convolution),
  * the image blurred in the FOURIER domain (IFFT(FFT(f) * FFT(h))),
  * the difference between the two (should be ~1e-13, i.e. floating-point noise),
  * the log-magnitude spectra of the image, the kernel and their product,
  * error metrics and timings.

Toggle "zero-pad before FFT" off to see the wrap-around error you get from
circular convolution.

To host the other assignments, add more routes/pages next to "/" and link them
in the navigation bar of templates/index.html.
"""

import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from blur_filters import (  # noqa: E402
    load_image, make_kernel, validate, to_uint8, fft_shape,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = {
    "cameraman": os.path.join(HERE, "sample", "cameraman.png"),
    "astronaut": os.path.join(HERE, "sample", "astronaut.png"),
}
MAX_SIDE = 512  # keeps the direct (spatial) convolution fast enough for a demo


def png_b64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def spectra_figure(img: np.ndarray, kernel: np.ndarray) -> str:
    """Log spectra of image, kernel, product + a 1-D slice of |H(u)| through the centre."""
    gray = img if img.ndim == 2 else img.mean(axis=2)
    H, W = gray.shape
    kh, kw = kernel.shape
    shape = fft_shape(H, W, kh, kw)  # same padded size used by the FFT filter
    F = np.fft.fft2(gray, s=shape)
    Hk = np.fft.fft2(kernel, s=shape)

    def show(x):
        return np.log1p(np.abs(np.fft.fftshift(x)))

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    ax[0].imshow(show(F), cmap="magma"); ax[0].set_title("log|F(u,v)|  image")
    ax[1].imshow(np.abs(np.fft.fftshift(Hk)), cmap="magma"); ax[1].set_title("|H(u,v)|  kernel")
    ax[2].imshow(show(F * Hk), cmap="magma"); ax[2].set_title("log|F(u,v)·H(u,v)|  product")
    for a in ax[:3]:
        a.axis("off")
    prof = np.abs(np.fft.fftshift(Hk))[shape[0] // 2, :]
    u = (np.arange(shape[1]) - shape[1] // 2) / shape[1]
    ax[3].plot(u, prof, lw=1.5)
    ax[3].set_title("|H(u,0)|  low-pass profile")
    ax[3].set_xlabel("frequency (cycles/pixel)"); ax[3].grid(alpha=.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html", samples=list(SAMPLES))


@app.route("/api/blur", methods=["POST"])
def api_blur():
    try:
        kind = request.form.get("kernel", "gaussian")
        k = int(request.form.get("size", 15))
        sigma_raw = request.form.get("sigma", "")
        sigma = float(sigma_raw) if sigma_raw not in ("", None) else None
        zero_pad = request.form.get("zero_pad", "true") == "true"
        if k % 2 == 0 or not (3 <= k <= 41):
            return jsonify(error="Kernel size must be an odd number between 3 and 41."), 400
        if sigma is not None and sigma <= 0:
            return jsonify(error="Sigma must be positive."), 400

        upload = request.files.get("image")
        if upload and upload.filename:
            tmp = io.BytesIO(upload.read())
            img = load_image(tmp, max_side=MAX_SIDE)
        else:
            name = request.form.get("sample", "cameraman")
            if name not in SAMPLES:
                return jsonify(error="Unknown sample image."), 400
            img = load_image(SAMPLES[name], max_side=MAX_SIDE)

        kernel = make_kernel(kind, k, sigma)
        res = validate(img, kernel, zero_pad=zero_pad)

        diff = np.abs(res["diff"])
        # amplify so the tiny difference is visible: scale by max (or leave black when ~0)
        dmax = diff.max()
        diff_vis = to_uint8(diff / dmax * 255) if dmax > 1e-6 else np.zeros_like(to_uint8(diff))
        if dmax <= 1e-6:
            diff_note = "Difference is at floating-point noise level (displayed as black)."
        else:
            diff_note = "Difference image is scaled so that its maximum appears white."

        m = res["metrics"]
        return jsonify(
            original=png_b64(to_uint8(img)),
            spatial=png_b64(to_uint8(res["spatial"])),
            fourier=png_b64(to_uint8(res["fourier"])),
            difference=png_b64(diff_vis),
            diff_note=diff_note,
            spectra=spectra_figure(img, kernel),
            kernel_preview=png_b64(to_uint8(kernel / kernel.max() * 255)),
            image_shape=list(img.shape),
            metrics=m,
            zero_pad=zero_pad,
        )
    except Exception as exc:  # report the problem to the page instead of a raw 500
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
