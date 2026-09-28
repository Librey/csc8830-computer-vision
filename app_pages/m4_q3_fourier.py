import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from webapp_utils import add_path, read_upload

Q3_DIR = add_path("module4", "q3")
import fourier_lab as fl  # noqa: E402

ASTRONAUT = os.path.join(os.path.dirname(Q3_DIR), "q1", "images", "astronaut.png")


def to_display(a, lo=1, hi=99.5):
    """Stretch a float array to 0-255 using percentiles, for viewing."""
    a = np.asarray(a, float)
    p0, p1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p0) / max(p1 - p0, 1e-12) * 255, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def run_filter(f, name, D0, order, sigma):
    return fl.filter_image(f, name, D0, order, sigma)


@st.cache_data(show_spinner=False)
def run_texture(noise, sigma):
    return fl.texture_demo(noise, sigma)


st.title("Q3: Edges and regions in the Fourier domain")
st.write("By the convolution theorem, filtering an image equals multiplying its spectrum "
         "by a transfer function $H(u,v)$. Edge detection comes down to choosing an $H$ "
         "that keeps high frequencies without amplifying noise.")

pdf_path = os.path.join(Q3_DIR, "q3_fourier_edges_segmentation.pdf")
if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as fh:
        st.download_button("Download the full derivation (PDF)", fh.read(),
                           "q3_fourier_edges_segmentation.pdf", "application/pdf")

tab_eq, tab_edges, tab_tex = st.tabs(["Key results", "Edge filter lab", "Texture segmentation lab"])

with tab_eq:
    st.markdown("**Derivative theorem.** Differentiating the inverse transform gives")
    st.latex(r"\mathcal{F}\left\{\frac{\partial f}{\partial x}\right\}=j2\pi u\,F(u,v),\qquad "
             r"\mathcal{F}\{\nabla^2 f\}=-4\pi^2(u^2+v^2)\,F(u,v)")
    st.markdown("Derivatives are high-pass filters. A step edge has $|F(u)|=A/(2\\pi|u|)$, "
                "so edges keep energy at high frequencies while smooth regions do not.")
    st.markdown("**Noise.** White noise has a flat spectrum, so a derivative amplifies it in "
                "proportion to frequency. A Gaussian low-pass caps the gain:")
    st.latex(r"G_\sigma(u,v)=e^{-2\pi^2\sigma^2(u^2+v^2)},\qquad "
             r"H_{\mathrm{LoG}}(\rho)=-4\pi^2\rho^2e^{-2\pi^2\sigma^2\rho^2}")
    st.latex(r"\frac{d}{d\rho}\,\rho^2e^{-2\pi^2\sigma^2\rho^2}=0\;\Rightarrow\;"
             r"\rho^*=\frac{1}{\sqrt2\,\pi\sigma}")
    st.markdown("The LoG is a band-pass whose peak $\\rho^*$ sets the edge scale; edges sit at "
                "its zero crossings. An ideal high-pass rings, because its kernel is a jinc, "
                "$D_0J_1(2\\pi D_0r)/r$, with side lobes; a Gaussian high-pass does not.")
    st.markdown("**Segmentation.** A Gabor filter is a Gaussian band-pass at $(u_0,v_0)$:")
    st.latex(r"h(x,y)=e^{-\frac{x^2+y^2}{2\sigma^2}}e^{j2\pi(u_0x+v_0y)}\;\Longleftrightarrow\;"
             r"H(u,v)=2\pi\sigma^2e^{-2\pi^2\sigma^2[(u-u_0)^2+(v-v_0)^2]}")
    st.markdown("Local energy in each band separates regions that differ in texture. The window "
                "obeys $\\Delta x\\,\\Delta u\\ge 1/(4\\pi)$, so sharper boundaries cost "
                "frequency resolution.")

with tab_edges:
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        src = st.radio("Image", ["Astronaut", "Upload your own"], horizontal=True, key="q3src")
        if src == "Astronaut":
            f = cv2.imread(ASTRONAUT, cv2.IMREAD_GRAYSCALE)
        else:
            up = st.file_uploader("Any image", type=["jpg", "jpeg", "png", "bmp"], key="q3up")
            if up is None:
                st.info("Upload an image, or switch back to the astronaut.")
                st.stop()
            img, _ = read_upload(up, max_side=512)
            f = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        name = st.selectbox("Filter H(u,v)", fl.FILTERS, index=4)
        D0, order, sigma = 0.05, 2, 2.0
        if name in ("Ideal high-pass", "Butterworth high-pass", "Gaussian high-pass",
                    "Gaussian low-pass"):
            D0 = st.slider("Cutoff D₀ (cycles/pixel)", 0.01, 0.30, 0.05, 0.01)
        if name == "Butterworth high-pass":
            order = st.slider("Order n", 1, 10, 2)
        if name in ("Gradient magnitude (Gaussian-smoothed)", "LoG band-pass"):
            sigma = st.slider("Gaussian σ (pixels)", 0.5, 8.0, 2.0, 0.5)
            st.caption(f"Peak frequency ρ* = 1/(√2·π·σ) = {1 / (np.sqrt(2) * np.pi * sigma):.3f} "
                       "cycles/pixel for the LoG.")
        freqs, prof = fl.radial_profile(name, D0, order, sigma)
        st.markdown("**|H| along one axis**")
        st.line_chart(pd.DataFrame({"|H|": prof / max(prof.max(), 1e-12)},
                                   index=pd.Index(np.round(freqs, 3), name="cycles/pixel")),
                      height=180)
    with c2:
        with st.spinner("FFT, multiply, inverse FFT..."):
            out = run_filter(f, name, D0, order, sigma)
        a, b = st.columns(2)
        a.image(to_display(out["spectrum"], 5, 99.9), caption="log(1 + |F(u,v)|), centred",
                width="stretch")
        b.image(to_display(out["H"], 0, 100), caption="|H(u,v)|, centred",
                width="stretch")
        a, b = st.columns(2)
        res = out["result"]
        if name == "LoG band-pass":
            a.image(to_display(res), caption="Filtered image (signed, gray = 0)",
                    width="stretch")
            b.image((~out["zero_crossings"]).astype(np.uint8) * 255,
                    caption="Zero crossings = edges", width="stretch")
        elif name == "Gaussian low-pass":
            a.image(f, caption="Input", width="stretch")
            b.image(to_display(res), caption="Low-passed: noise and fine detail removed",
                    width="stretch")
        else:
            a.image(f, caption="Input", width="stretch")
            b.image(to_display(np.abs(res), 0, 99.5), caption="|filtered image|",
                    width="stretch")
        if name == "Ideal high-pass":
            st.caption("Look for faint ripples running parallel to strong edges: that is the "
                       "ringing from the ideal filter's sharp cutoff. Switch to the Gaussian "
                       "high-pass with the same D₀ and it disappears.")

with tab_tex:
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.write("An ellipse of oblique stripes (0.06 cycles/px, 60°) sits inside vertical "
                 "stripes (0.10 cycles/px). Both have the same mean brightness, so a "
                 "threshold cannot separate them. Two Gabor band-passes can.")
        noise = st.slider("Noise level (stripe amplitude = 1)", 0.0, 2.0, 0.6, 0.1)
        gsig = st.slider("Gabor σ (pixels)", 2.0, 16.0, 6.0, 1.0,
                         help="Large σ: narrow band, better texture discrimination, "
                              "blurrier boundary. Small σ: the reverse.")
        d = run_texture(noise, gsig)
        st.metric("Pixels labelled correctly", f"{d['accuracy'] * 100:.1f}%")
    with c2:
        a, b = st.columns(2)
        a.image(to_display(d["image"], 0, 100), caption="Input", width="stretch")
        b.image(to_display(d["spectrum"], 5, 99.9),
                caption="Spectrum: one bright pair of peaks per texture",
                width="stretch")
        a, b = st.columns(2)
        lr = d["log_ratio"]
        lim = np.percentile(np.abs(lr), 99)
        lr_img = cv2.applyColorMap(np.clip((lr / lim + 1) * 127.5, 0, 255).astype(np.uint8),
                                   cv2.COLORMAP_COOL)
        a.image(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB),
                caption="log E_B − log E_A (Gabor energy ratio)", width="stretch")
        vis = cv2.cvtColor(to_display(d["image"], 0, 100), cv2.COLOR_GRAY2BGR)
        for m, col in [(d["truth"], (0, 200, 0)), (d["segmentation"], (0, 0, 255))]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, col, 2)
        b.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                caption="Found boundary (red) against the true one (green)",
                width="stretch")
