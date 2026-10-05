import io
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from webapp_utils import ROOT, add_path, read_upload

add_path("module6", "part_a")
import lk_validate as lk  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "module6", "web_samples")
SAMPLES = {
    "Traffic, frames 599 → 600": dict(stem="Traffic_static_shot_t20", moving_only=True,
                                       band=(0.0, 1.0)),
    "Dashcam, frames 625 → 626": dict(stem="Driving_pov_t25", moving_only=False,
                                       band=(0.42, 0.72)),
}


@st.cache_data(show_spinner=False)
def load_pair(stem):
    a = cv2.imread(os.path.join(SAMPLE_DIR, stem + "_a.png"))
    b = cv2.imread(os.path.join(SAMPLE_DIR, stem + "_b.png"))
    return a, b


@st.cache_data(show_spinner=False)
def run(A, B, n, moving_only, band):
    rows, I, J = lk.validate_pair(A, B, n, moving_only, band[0], band[1])
    return rows, I, J


@st.cache_data(show_spinner=False)
def hand_example(I, J, x, y):
    """The 5x5, single-level calculation from the report, for one point."""
    buf = io.StringIO()
    wk = lk.worked_example(I, J, (x, y), buf)
    Ix, Iy = lk.gradients(I)
    _, hist, _ = lk.lk_level(I, J, Ix, Iy, np.array([float(round(x)), float(round(y))]),
                             np.zeros(2), 2)
    return wk, buf.getvalue(), hist


def overview(img, rows):
    """Frame t with each tracked point numbered and its motion drawn 5x longer."""
    vis = img.copy()
    t = max(1, int(round(img.shape[1] / 500)))
    for r in rows:
        p = (int(round(r["x"])), int(round(r["y"])))
        q = (int(round(r["x"] + 5 * r["my_dx"])), int(round(r["y"] + 5 * r["my_dy"])))
        cv2.arrowedLine(vis, p, q, (0, 255, 255), t, tipLength=0.25)
        cv2.circle(vis, p, 4 * t, (0, 255, 255), t)
        cv2.putText(vis, str(r["id"]), (p[0] + 6 * t, p[1] - 6 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * t, (255, 255, 255), t + 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def crop(img, cx, cy, half=18, zoom=5):
    """Sub-pixel crop centred on (cx, cy), enlarged, with a crosshair on the centre."""
    M = np.float32([[1, 0, half - cx], [0, 1, half - cy]])
    c = cv2.warpAffine(img, M, (2 * half + 1, 2 * half + 1), flags=cv2.INTER_LINEAR)
    c = cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    m = half * zoom + zoom // 2
    cv2.line(c, (m, 0), (m, c.shape[0]), (255, 255, 0), 1)
    cv2.line(c, (0, m), (c.shape[1], m), (255, 255, 0), 1)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


st.title("Lucas–Kanade tracking, checked against real pixels")
st.write("Pick two consecutive frames. The page finds strong corners in the first frame and "
         "tracks each one into the second frame three independent ways: the Lucas–Kanade "
         "equations derived in the report (coded from scratch with bilinear interpolation), "
         "OpenCV's `calcOpticalFlowPyrLK`, and a direct patch match that measures where the "
         "point actually went.")

left, right = st.columns([1, 2.4], gap="large")
with left:
    source = st.radio("Frames", ["Report frame pairs", "Upload two frames"], horizontal=True)
    if source == "Report frame pairs":
        name = st.selectbox("Pair", list(SAMPLES))
        cfg = SAMPLES[name]
        A, B = load_pair(cfg["stem"])
        moving_default, band_default = cfg["moving_only"], cfg["band"]
    else:
        ua = st.file_uploader("Frame t", type=["png", "jpg", "jpeg", "bmp"])
        ub = st.file_uploader("Frame t+1", type=["png", "jpg", "jpeg", "bmp"])
        if ua is None or ub is None:
            st.info("Upload two consecutive frames of the same size. PNG keeps every pixel "
                    "exact; JPEG compression adds a little noise.")
            st.stop()
        A, _ = read_upload(ua, max_side=1000)
        B, _ = read_upload(ub, max_side=1000)
        if A is None or B is None or A.shape != B.shape:
            st.error("Both frames must be images of the same size.")
            st.stop()
        if A.ndim == 2:
            A, B = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR), cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
        moving_default, band_default = False, (0.0, 1.0)

    n = st.slider("Points to track", 3, 12, 6)
    moving_only = st.checkbox("Only corners on moving objects", value=moving_default,
                              help="Useful with a fixed camera: skips the still background.")
    band = st.slider("Search rows (fraction of image height)", 0.0, 1.0, band_default, 0.01,
                     help="For the dashcam pair, 0.42 to 0.72 keeps the roadside and skips the "
                          "dashboard and sky.")

with st.spinner("Tracking..."):
    rows, I, J = run(A, B, n, moving_only, tuple(band))
if not rows:
    st.warning("No trackable corners in that region. Widen the row band or untick the "
               "moving-objects option.")
    st.stop()

with left:
    st.image(overview(A, rows), caption="Tracked points in frame t; arrows show the motion, "
                                         "drawn 5× longer")

with right:
    err = np.array([r["err_meas"] for r in rows])
    errc = np.array([r["err_cv"] for r in rows])
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean error vs measured", f"{err.mean():.3f} px")
    m2.metric("Max error vs measured", f"{err.max():.3f} px")
    m3.metric("Mean difference from OpenCV", f"{errc.mean():.3f} px")
    table = pd.DataFrame([{
        "Pt": r["id"], "p (frame t)": f"({r['x']:.0f}, {r['y']:.0f})",
        "Motion d": f"({r['my_dx']:.2f}, {r['my_dy']:.2f})",
        "Predicted p′": f"({r['my_x']:.2f}, {r['my_y']:.2f})",
        "Measured p′": f"({r['meas_x']:.2f}, {r['meas_y']:.2f})",
        "Error vs measured (px)": round(r["err_meas"], 3),
        "Error vs OpenCV (px)": round(r["err_cv"], 3),
        "Iterations": r["iters"], "Patch match score": round(r["ncc"], 2)} for r in rows])
    st.dataframe(table, hide_index=True, width="stretch")

    st.caption("Top row: frame t centred on p. Bottom row: frame t+1 centred on the predicted "
               "p′. When tracking works, the same feature sits under the crosshair in both.")
    cols = st.columns(len(rows))
    for c, r in zip(cols, rows):
        c.image(crop(A, r["x"], r["y"]), caption=f"pt {r['id']}")
        c.image(crop(B, r["my_x"], r["my_y"]), caption=f"{r['err_meas']:.2f} px off")

st.subheader("One point by hand")
st.write("The same calculation as the report's worked example: a 5×5 window, central "
         "differences for the gradients, the 2×2 system G d = e solved with Cramer's rule, then "
         "Newton iterations that re-sample frame t+1 with bilinear interpolation.")
pick = st.selectbox("Point", [r["id"] for r in rows],
                    format_func=lambda i: f"pt {i} at ({rows[i-1]['x']:.0f}, {rows[i-1]['y']:.0f})")
r = rows[pick - 1]
wk, text, hist = hand_example(I, J, r["x"], r["y"])
h1, h2 = st.columns([1.3, 1], gap="large")
with h1:
    st.code(text.split("sum Ix^2")[0].strip(), language=None)
with h2:
    st.latex(r"G=\begin{bmatrix}%.1f & %.1f\\ %.1f & %.1f\end{bmatrix},\quad "
             r"\mathbf{e}=\begin{bmatrix}%.1f\\ %.1f\end{bmatrix}"
             % (wk["sxx"], wk["sxy"], wk["sxy"], wk["syy"], -wk["sxt"], -wk["syt"]))
    st.latex(r"\det G = %.4g,\quad \lambda = %.0f,\ %.0f" % (wk["det"], wk["lam"][0], wk["lam"][1]))
    st.latex(r"u_1=\frac{-\sum I_y^2\sum I_xI_t+\sum I_xI_y\sum I_yI_t}{\det G}=%.4f" % wk["u"])
    st.latex(r"v_1=\frac{\sum I_xI_y\sum I_xI_t-\sum I_x^2\sum I_yI_t}{\det G}=%.4f" % wk["v"])
    st.dataframe(pd.DataFrame([{"Iteration": k, "u (px)": round(u, 4), "v (px)": round(v, 4),
                                "Step size (px)": round(s, 4)} for k, u, v, s in hist]),
                 hide_index=True, width="stretch")
    meas = (r["meas_x"] - r["x"], r["meas_y"] - r["y"])
    st.caption(f"Measured displacement: ({meas[0]:.3f}, {meas[1]:.3f}). Without a pyramid the "
               "5×5 window only handles motions of a few pixels, so very fast points may not "
               "converge here even when the full tracker above gets them right.")
