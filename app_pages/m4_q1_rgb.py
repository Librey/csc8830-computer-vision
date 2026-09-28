import os

import cv2
import pandas as pd
import streamlit as st

from webapp_utils import (add_path, draw_prompt, metrics_rows, overlay_mask, parse_points,
                          png_bytes, read_mask, read_upload, rgb)

Q1_DIR = add_path("module4", "q1")
import q1_rgb_human_boundary as q1  # noqa: E402

# Sample photos with the box and clicks used in the report
SAMPLES = {
    "Astronaut (plain background)": dict(file="astronaut.png", sam="astronaut_sam2.png",
                                         rect=(15, 5, 360, 507), points="", radius=45),
    "Messi (grass and crowd)": dict(file="messi5.jpg", sam="messi5_sam2.png",
                                    rect=(65, 55, 395, 287),
                                    points="110,300; 170,320; 360,310", radius=25),
    "Zidane (two people in black suits)": dict(file="zidane.jpg", sam="zidane_sam2.png",
                                               rect=(115, 195, 1045, 525),
                                               points="860,300; 850,520; 880,680; 950,620; "
                                                      "1000,400; 1050,500; 1060,650; 820,420",
                                               radius=45),
}


@st.cache_data(show_spinner=False)
def load_sample(name):
    s = SAMPLES[name]
    img = cv2.imread(os.path.join(Q1_DIR, "images", s["file"]))
    sam = cv2.imread(os.path.join(Q1_DIR, "sam2_masks", s["sam"]), cv2.IMREAD_GRAYSCALE)
    return img, sam


@st.cache_data(show_spinner=False)
def segment(img, rect, method, points, radius):
    if method == "GrabCut":
        return q1.segment_grabcut(img, rect, bg_points=list(points) or None, radius=radius)
    return q1.segment_watershed(img, rect)


st.title("Q1: Human boundary in an RGB photo")
st.write("Set a box around the person and the script finds the exact outline with "
         "classical OpenCV methods only. For the sample photos, the page also compares "
         "the result with a SAM2 mask of the same person.")

left, right = st.columns([1, 2], gap="large")

with left:
    source = st.radio("Image", ["Sample photo", "Upload your own"], horizontal=True)
    sam_mask = None
    if source == "Sample photo":
        name = st.selectbox("Sample", list(SAMPLES))
        img, sam_mask = load_sample(name)
        defaults = SAMPLES[name]
        key = name
    else:
        up = st.file_uploader("RGB photo of a person", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is None:
            st.info("Upload a photo to start. Large photos are shrunk to 1280 px on the long side.")
            st.stop()
        img, _ = read_upload(up)
        if img is None:
            st.error("Could not read that file as an image.")
            st.stop()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        H, W = img.shape[:2]
        mx, my = int(0.05 * W), int(0.05 * H)
        defaults = dict(rect=(mx, my, W - 2 * mx, H - 2 * my), points="", radius=45)
        key = up.name + str(img.shape)

    H, W = img.shape[:2]
    method = st.radio("Method", ["GrabCut", "Watershed"], horizontal=True,
                      help="GrabCut: colour models plus a graph cut. "
                           "Watershed: fully classical flooding from seeds.")

    st.markdown(f"**Box around the person** (image is {W} × {H} px)")
    dx, dy, dw, dh = defaults["rect"]
    c1, c2 = st.columns(2)
    x = c1.number_input("x", 0, W - 2, min(dx, W - 2), key=f"x{key}")
    y = c2.number_input("y", 0, H - 2, min(dy, H - 2), key=f"y{key}")
    w = c1.number_input("width", 2, W - x, max(2, min(dw, W - x)), key=f"w{key}")
    h = c2.number_input("height", 2, H - y, max(2, min(dh, H - y)), key=f"h{key}")
    rect = (int(x), int(y), int(w), int(h))

    points, radius = [], defaults["radius"]
    if method == "GrabCut":
        pts_text = st.text_area("Background clicks (optional)", defaults["points"],
                                key=f"p{key}", height=80,
                                help="Points that are NOT the person, as x,y; x,y. "
                                     "This works like SAM2's Remove click.")
        points = parse_points(pts_text)
        radius = st.slider("Click radius (px)", 5, 100, defaults["radius"], key=f"r{key}")

    if source == "Upload your own":
        sam_up = st.file_uploader("SAM2 mask to compare against (optional)", type=["png", "jpg"],
                                  help="Black and white, person in white, same size as the photo.")
        if sam_up is not None:
            sam_mask = read_mask(sam_up, img.shape)

    run = st.button("Find boundary", type="primary", width="stretch")

signature = (key, rect, method, tuple(points), radius)
if run:
    st.session_state["q1_sig"] = signature

with right:
    if st.session_state.get("q1_sig") != signature:
        st.image(rgb(draw_prompt(img, rect, points, radius if points else 0)),
                 caption="Your box (yellow) and background clicks (red). "
                         "Press Find boundary to run.", width="stretch")
    else:
        with st.spinner("Segmenting..."):
            mask = segment(img, rect, method, tuple(points), radius)
        tabs = ["Boundary", "Mask"] + (["Compare with SAM2"] if sam_mask is not None else [])
        t = st.tabs(tabs)
        with t[0]:
            st.image(rgb(overlay_mask(img, mask)), width="stretch")
        with t[1]:
            st.image(mask, width="stretch", clamp=True)
        if sam_mask is not None:
            with t[2]:
                st.image(rgb(q1.make_comparison_image(img, mask, sam_mask)),
                         caption="Green: both agree. Red: only ours. Blue: only SAM2.",
                         width="stretch")
            m = q1.compare_masks(mask, sam_mask)
            st.markdown("**Agreement with SAM2**")
            st.dataframe(pd.DataFrame(metrics_rows(m)), hide_index=True, width="stretch")
        st.download_button("Download mask (PNG)", png_bytes(mask), "mask.png", "image/png")

with st.expander("How the methods work"):
    st.markdown(
        "- **GrabCut** fits colour models to the pixels inside and outside your box, then "
        "solves a min-cut on the pixel graph. Background clicks mark pixels as definite "
        "background and rerun the cut. It fails when the person and background share "
        "colours, like the two black suits in the Zidane photo.\n"
        "- **Watershed** scores each pixel by how often its colour appears inside the box "
        "compared with outside, thresholds that score with Otsu's method to seed the "
        "person and background, then floods from both seeds and keeps the line where "
        "they meet.\n"
        "- Both methods finish with morphological cleanup, keep the large connected "
        "regions, fill holes and trace the outline with `cv2.findContours`.")
