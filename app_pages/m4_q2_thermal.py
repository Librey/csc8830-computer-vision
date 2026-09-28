import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from webapp_utils import (add_path, draw_prompt, metrics_rows, overlay_mask, png_bytes,
                          read_mask, read_upload, rgb)

Q2_DIR = add_path("module4", "q2")
import q2_thermal_human_boundary as q2  # noqa: E402

SAMPLES = {
    "010025: cold night background": dict(stem="llvip_010025", rect=(465, 365, 115, 260)),
    "200007: warm pavement, post behind him": dict(stem="llvip_200007", rect=(1030, 495, 175, 330)),
    "190003: warm pavement, cold jacket hem": dict(stem="llvip_190003", rect=(350, 685, 110, 290)),
}


@st.cache_data(show_spinner=False)
def load_sample(name, colour):
    stem = SAMPLES[name]["stem"]
    fname = f"{stem}_jet.png" if colour else f"{stem}.jpg"
    img = cv2.imread(os.path.join(Q2_DIR, "images", fname))
    sam = cv2.imread(os.path.join(Q2_DIR, "sam2_masks", f"{stem}_sam2.png"), cv2.IMREAD_GRAYSCALE)
    return img, sam


@st.cache_data(show_spinner=False)
def heat_map(img, black_hot, palette):
    return q2.to_gray(img, black_hot, palette)


@st.cache_data(show_spinner=False)
def segment(gray, rect, method):
    return q2.segment_thermal(gray, rect, method)


st.title("Q2: Human boundary in a thermal image")
st.write("A thermal camera measures heat, so a person shows up warmer than a cold "
         "background whatever they wear. The script separates warm, compact shapes from "
         "the background with classical morphology and thresholding only.")

left, right = st.columns([1, 2], gap="large")

with left:
    source = st.radio("Image", ["Sample image", "Upload your own"], horizontal=True)
    sam_mask = None
    if source == "Sample image":
        name = st.selectbox("Sample (LLVIP dataset)", list(SAMPLES))
        colour = st.toggle("Rainbow palette", value=True,
                           help="Off shows the camera's grayscale heat values. On shows the "
                                "same data painted with the jet palette, which the script "
                                "decodes back to heat.")
        img, sam_mask = load_sample(name, colour)
        default_rect = SAMPLES[name]["rect"]
        key = name
    else:
        up = st.file_uploader("Thermal image (grayscale or false colour)",
                              type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        if up is None:
            st.info("Upload a thermal image to start. Large images are shrunk to 1280 px.")
            st.stop()
        img, _ = read_upload(up)
        if img is None:
            st.error("Could not read that file as an image.")
            st.stop()
        H, W = img.shape[:2]
        default_rect = (0, 0, W, H)
        key = up.name + str(img.shape)

    H, W = img.shape[:2]
    method = st.radio("Method", ["Top-hat + hysteresis", "Plain Otsu + hysteresis"],
                      horizontal=True)
    method_key = "tophat" if method.startswith("Top") else "otsu"

    with st.expander("Camera settings"):
        black_hot = st.checkbox("Black-hot camera (warm = dark)", key=f"bh{key}")
        palette = st.selectbox("Palette of a colour image", ["auto"] + list(q2.PALETTES),
                               key=f"pal{key}")

    st.markdown(f"**Box around the person** (image is {W} × {H} px)")
    dx, dy, dw, dh = default_rect
    c1, c2 = st.columns(2)
    x = c1.number_input("x", 0, W - 2, min(dx, W - 2), key=f"x{key}")
    y = c2.number_input("y", 0, H - 2, min(dy, H - 2), key=f"y{key}")
    w = c1.number_input("width", 2, W - x, max(2, min(dw, W - x)), key=f"w{key}")
    h = c2.number_input("height", 2, H - y, max(2, min(dh, H - y)), key=f"h{key}")
    rect = (int(x), int(y), int(w), int(h))

    if source == "Upload your own":
        sam_up = st.file_uploader("SAM2 mask to compare against (optional)", type=["png", "jpg"])
        if sam_up is not None:
            sam_mask = read_mask(sam_up, img.shape)

    run = st.button("Find boundary", type="primary", width="stretch")

signature = (key, source == "Sample image" and colour, rect, method_key, black_hot, palette)
if run:
    st.session_state["q2_sig"] = signature

with right:
    if st.session_state.get("q2_sig") != signature:
        st.image(rgb(draw_prompt(img, rect)), width="stretch",
                 caption="Your box in yellow. Press Find boundary to run.")
    else:
        with st.spinner("Decoding heat and segmenting..."):
            gray = heat_map(img, black_hot, palette)
            mask = segment(gray, rect, method_key)
        is_colour = img.ndim == 3 and q2.is_false_colour(img)
        base = img if is_colour else q2.false_colour(gray)
        zoom = st.toggle("Zoom to the box", value=True,
                         help="People in thermal scenes are often small; zooming makes the "
                              "boundary visible.")
        x0, y0, w0, h0 = rect
        mg = int(0.15 * max(w0, h0))
        cx, cy = max(0, x0 - mg), max(0, y0 - mg)
        crop = (cx, cy, min(W, x0 + w0 + mg) - cx, min(H, y0 + h0 + mg) - cy)

        def view(a):
            if not zoom:
                return a
            X, Y, CW, CH = crop
            return a[Y:Y + CH, X:X + CW]
        tabs = ["Boundary", "Heat values", "Mask"] + (["Compare with SAM2"] if sam_mask is not None else [])
        t = st.tabs(tabs)
        with t[0]:
            st.image(rgb(overlay_mask(view(base), view(mask), (255, 255, 255))),
                     width=460 if zoom else "stretch")
        with t[1]:
            cap = "Heat recovered from the palette colours (white = hot)." if is_colour \
                else "Heat values as the camera stored them (white = hot)."
            st.image(view(gray), width=460 if zoom else "stretch", caption=cap)
            if is_colour:
                naive = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(view(naive), width=460 if zoom else "stretch",
                         caption="For contrast: a plain grayscale conversion of the colour "
                                 "image. Green comes out brighter than red, so hot and warm "
                                 "get scrambled.")
        with t[2]:
            st.image(view(mask), width=460 if zoom else "stretch", clamp=True)
        if sam_mask is not None:
            with t[3]:
                comp = q2.make_comparison_image(gray, mask, sam_mask, crop,
                                                base=img.copy() if is_colour else None)
                st.image(rgb(comp), width="stretch",
                         caption="Zoomed on the box. Green: both agree. Red: only ours. "
                                 "Blue: only SAM2.")
            st.markdown("**Agreement with SAM2**")
            st.dataframe(pd.DataFrame(metrics_rows(q2.compare_masks(mask, sam_mask))),
                         hide_index=True, width="stretch")
        st.download_button("Download mask (PNG)", png_bytes(mask), "thermal_mask.png", "image/png")

with st.expander("How the method works"):
    st.markdown(
        "1. **Heat values.** Grayscale input is used as is. A false-colour image is decoded "
        "by matching each pixel's colour to its position on the palette.\n"
        "2. **Bilateral filter** smooths sensor noise but keeps the body edge sharp.\n"
        "3. **White top-hat**, T = I − open(I, B) with a disc B wider than the person, "
        "removes slowly varying background heat such as warm pavement.\n"
        "4. **Hysteresis threshold.** Otsu's threshold marks sure-person pixels; cooler "
        "pixels above a lower threshold join only if they connect to them. This recovers "
        "insulated clothing.\n"
        "5. **Cleanup**: a closing sized to the body bridges cold belts and collars, then "
        "holes are filled and `cv2.findContours` traces the boundary.")
