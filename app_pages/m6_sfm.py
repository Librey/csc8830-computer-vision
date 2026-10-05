import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from webapp_utils import ROOT, add_path, rgb

PART_B = add_path("module6", "part_b")
import sfm_book  # noqa: E402

IMG_DIR = os.path.join(PART_B, "images")
CAPTIONS = ["View 1: overhead", "View 2: book sideways", "View 3: spine side",
            "View 4: from the bottom edge"]
TRUE_F = 24


@st.cache_data(show_spinner=False)
def thumbs():
    out = []
    for v in sfm_book.VIEWS:
        im = cv2.imread(os.path.join(IMG_DIR, v + ".jpeg"))
        out.append(rgb(cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)))
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def reconstruct(f_eq, width, length):
    """Runs the full pipeline from sfm_book.py and keeps its numbers and figures."""
    out = tempfile.mkdtemp(prefix="sfm_")
    res = sfm_book.run(IMG_DIR, out, f_eq=f_eq, width=width, length=length, verbose=False)
    figs = {}
    for k in ("fig_corners", "fig_3d", "fig_matches"):
        with open(os.path.join(out, k + ".png"), "rb") as f:
            figs[k] = f.read()
    with open(os.path.join(out, "results.txt")) as f:
        res["log"] = f.read()
    res["figs"] = figs
    return res


st.title("Structure from motion: a book from four photos")
st.write("Four iPhone photos of a 14.5 × 20 cm book lying flat. The pipeline matches SIFT "
         "features on the cover, fits a homography between view 1 and each other view, "
         "decomposes each homography into the camera's rotation and translation, triangulates "
         "the corners, and refines everything with bundle adjustment. Only the 14.5 cm width "
         "sets the scale; the length and the right angles are left as checks.")

cols = st.columns(4)
for c, im, cap in zip(cols, thumbs(), CAPTIONS):
    c.image(im, caption=cap)

st.subheader("Try a different focal length")
st.write("The photos were taken at 24 mm (35 mm equivalent). Tell the pipeline something else "
         "and watch the reconstruction bend: the cover stops being a rectangle, the two width "
         "edges disagree, and the reprojection error grows. That is the reason camera "
         "calibration matters.")
c1, c2, c3 = st.columns([2, 1, 1])
f_eq = c1.select_slider("Focal length assumed (35 mm equivalent, mm)",
                        options=list(range(18, 31)), value=TRUE_F)
width = c2.number_input("Cover width (cm)", 5.0, 50.0, 14.5, 0.1)
length = c3.number_input("Cover length (cm)", 5.0, 50.0, 20.0, 0.1)

with st.spinner("Reconstructing (about 20 seconds the first time for each setting)..."):
    res = reconstruct(float(f_eq), float(width), float(length))
    base = reconstruct(float(TRUE_F), float(width), float(length)) if f_eq != TRUE_F else res

sides, angs = res["sides"], res["angles"]
L = (sides[1] + sides[3]) / 2
m1, m2, m3, m4 = st.columns(4)
m1.metric("Focal length in pixels", f"{res['f_px']:.0f} px")
m2.metric("Recovered length", f"{L:.2f} cm", f"{L - length:+.2f} cm vs true", delta_color="off")
m3.metric("Worst corner angle", f"{max(abs(a - 90) for a in angs):.2f}° off 90°",
          None if f_eq == TRUE_F else
          f"{max(abs(a - 90) for a in angs) - max(abs(a - 90) for a in base['angles']):+.2f}° vs 24 mm",
          delta_color="inverse")
m4.metric("Max reprojection error", f"{res['reproj_max']:.1f} px",
          None if f_eq == TRUE_F else f"{res['reproj_max'] - base['reproj_max']:+.1f} px vs 24 mm",
          delta_color="inverse")

left, right = st.columns([1, 1.4], gap="large")
with left:
    st.markdown("**Recovered cover**")
    diag_true = float(np.hypot(width, length))
    st.dataframe(pd.DataFrame([
        {"Measurement": "Width TL–TR, BR–BL (cm)", "Recovered": f"{sides[0]:.2f}, {sides[2]:.2f}",
         "True": f"{width:.1f}"},
        {"Measurement": "Length TR–BR, BL–TL (cm)", "Recovered": f"{sides[1]:.2f}, {sides[3]:.2f}",
         "True": f"{length:.1f}"},
        {"Measurement": "Diagonals (cm)",
         "Recovered": f"{res['diagonals'][0]:.2f}, {res['diagonals'][1]:.2f}",
         "True": f"{diag_true:.2f}"},
        {"Measurement": "Corner angles TL, TR, BR, BL (°)",
         "Recovered": ", ".join(f"{a:.1f}" for a in angs), "True": "90"},
    ]), hide_index=True, width="stretch")
    st.markdown("**Camera positions** (origin at the bottom of the spine, x along the bottom "
                "edge, y up the spine, z height above the cover)")
    st.dataframe(pd.DataFrame([{
        "View": i + 1, "x (cm)": round(float(c["position"][0]), 1),
        "y (cm)": round(float(c["position"][1]), 1), "z (cm)": round(float(c["position"][2]), 1),
        "Tilt from straight down (°)": round(c["tilt"], 1)} for i, c in enumerate(res["cameras"])]),
        hide_index=True, width="stretch")
    st.caption(f"Bundle adjustment: mean reprojection error {res['reproj_mean']:.2f} px. "
               f"SIFT inliers per view pair: "
               + ", ".join(str(v) for v in res["inliers"].values()) + ".")
with right:
    st.image(res["figs"]["fig_3d"], caption="Recovered cover, surface points and cameras (left); "
                                            "top view against the true rectangle (right)")

st.image(res["figs"]["fig_corners"], caption="Corners in each view (cyan) and the 3D corners "
                                             "projected back into each view (red +)")
with st.expander("SIFT matches between view 1 and the other views"):
    st.image(res["figs"]["fig_matches"])
with st.expander("Full log from sfm_book.py (every matrix and intermediate number)"):
    st.code(res["log"], language=None)
