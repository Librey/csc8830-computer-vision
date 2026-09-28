import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from webapp_utils import add_path

M3_DIR = add_path("module3")
import blur_filters as bf  # noqa: E402

SAMPLES = {"Cameraman": "cameraman.png", "Astronaut": "astronaut.png"}
MAX_SIDE = 512  # keeps the direct spatial convolution fast enough for a demo


def to_display(a, lo=1, hi=99.8):
    a = np.asarray(a, float)
    p0, p1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p0) / max(p1 - p0, 1e-12) * 255, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def load(sample, upload_bytes):
    if upload_bytes is not None:
        return bf.load_image(io.BytesIO(upload_bytes), max_side=MAX_SIDE)
    return bf.load_image(os.path.join(M3_DIR, "sample", SAMPLES[sample]), max_side=MAX_SIDE)


@st.cache_data(show_spinner=False)
def blur(img, kind, k, sigma, zero_pad):
    kernel = bf.make_kernel(kind, k, sigma)
    return kernel, bf.validate(img, kernel, zero_pad=zero_pad)


st.title("Module 3: Blur in space, blur in frequency")
st.write("The same box or Gaussian blur, computed two ways: direct convolution in the "
         "spatial domain, and multiplication of FFTs in the Fourier domain. By the "
         "convolution theorem they should agree to floating-point rounding.")

left, right = st.columns([1, 2], gap="large")
with left:
    source = st.radio("Image", ["Sample image", "Upload your own"], horizontal=True)
    upload_bytes, sample = None, "Cameraman"
    if source == "Sample image":
        sample = st.selectbox("Sample", list(SAMPLES))
    else:
        up = st.file_uploader("Any image", type=["png", "jpg", "jpeg", "bmp"])
        if up is None:
            st.info("Upload an image to start. It is shrunk to 512 px on the long side.")
            st.stop()
        upload_bytes = up.getvalue()
    kind = st.radio("Kernel", ["gaussian", "box"], horizontal=True,
                    format_func=lambda s: s.capitalize())
    k = st.slider("Kernel size K (odd)", 3, 41, 15, 2)
    sigma = None
    if kind == "gaussian":
        auto = st.checkbox("Pick σ from K automatically", value=False)
        if not auto:
            sigma = st.slider("σ (pixels)", 0.5, 10.0, 2.5, 0.5)
    zero_pad = st.toggle("Zero-pad before the FFT", value=True,
                         help="Off: the DFT does circular convolution, so the blur wraps "
                              "around the image edges.")

img = load(sample, upload_bytes)
with st.spinner("Convolving in space and in frequency..."):
    kernel, res = blur(img, kind, k, sigma, zero_pad)
m = res["metrics"]

with right:
    a, b, c = st.columns(3)
    a.image(bf.to_uint8(res["spatial"]), caption="Spatial domain (direct convolution)",
            width="stretch")
    b.image(bf.to_uint8(res["fourier"]), caption="Fourier domain (IFFT of F·H)",
            width="stretch")
    diff = np.abs(res["diff"])
    dmax = diff.max()
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    if dmax > 1e-6:
        c.image((diff / dmax * 255).astype(np.uint8), width="stretch",
                caption="|difference|, scaled so its maximum is white")
    else:
        c.image(np.zeros(diff.shape, np.uint8), width="stretch",
                caption="|difference| is at rounding level, so it shows as black")

    if zero_pad:
        st.success(f"Largest difference: {m['max_abs_error']:.2e} grey levels. "
                   "Both methods compute the same blur.")
    else:
        st.warning(f"Largest difference: {m['max_abs_error']:.2f} grey levels. Without "
                   "padding the FFT wraps the blur around the edges; the error sits "
                   "along the borders.")

    gray = img if img.ndim == 2 else img.mean(axis=2)
    shape = bf.fft_shape(gray.shape[0], gray.shape[1], *kernel.shape)
    F = np.fft.fft2(gray, s=shape)
    Hk = np.fft.fft2(kernel, s=shape)
    s1, s2, s3 = st.columns(3)
    s1.image(to_display(np.log1p(np.abs(np.fft.fftshift(F)))), caption="log|F(u,v)|, image",
             width="stretch")
    s2.image(to_display(np.abs(np.fft.fftshift(Hk)), 0, 100), caption="|H(u,v)|, kernel",
             width="stretch")
    s3.image(to_display(np.log1p(np.abs(np.fft.fftshift(F * Hk)))),
             caption="log|F·H|, blurred image", width="stretch")

    st.dataframe(pd.DataFrame([
        {"Measure": "Max |spatial − Fourier|", "Value": f"{m['max_abs_error']:.3e}"},
        {"Measure": "RMSE", "Value": f"{m['rmse']:.3e}"},
        {"Measure": "Max |spatial − SciPy reference|", "Value": f"{m['spatial_vs_scipy_max_abs']:.3e}"},
        {"Measure": "Kernel sum", "Value": f"{m['kernel_sum']:.6f}"},
        {"Measure": "Spatial time (s)", "Value": f"{m['t_spatial_s']:.3f}"},
        {"Measure": "FFT time (s)", "Value": f"{m['t_fft_s']:.3f}"},
    ]), hide_index=True, width="stretch")

st.caption("Module 3 also has its original Flask app: `cd module3` then `python app.py`.")
