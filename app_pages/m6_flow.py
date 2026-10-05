import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from webapp_utils import ROOT, add_path

add_path("module6", "part_a")
from optical_flow import draw_arrows, flow_to_hsv  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "module6", "web_samples")
SAMPLES = {
    "Traffic from an overpass (camera fixed)": "traffic_clip.mp4",
    "Dashcam on a desert road (camera moving)": "driving_clip.mp4",
}
MAX_W = 640          # flow is computed at this width at most
VIEW_W = 560         # width of each panel shown on the page
MAX_SECONDS = 8


def colour_wheel(size=110):
    """Legend for the HSV coding: direction -> hue, distance from centre -> speed."""
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    flow = np.dstack([x, y]).astype(np.float32)
    img = flow_to_hsv(flow, max_mag=1.0)
    img[np.hypot(x, y) > 1] = 255
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False, max_entries=4)
def process(video_bytes, start, seconds):
    """Farneback flow for every frame pair (same settings as optical_flow.py).
    Keeps per-frame statistics for all frames and display panels for ~60 of them."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    n = int(seconds * fps)
    keep_every = max(1, n // 60)
    ok, prev = cap.read()
    if not ok:
        cap.release(); os.unlink(path)
        return None
    s = min(1.0, MAX_W / prev.shape[1])
    prev = cv2.resize(prev, None, fx=s, fy=s)
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    stats, panels = [], []
    for i in range(1, n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, None, fx=s, fy=s)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        moving = mag > 1.0
        mdx = float(flow[..., 0][moving].mean()) if moving.any() else 0.0
        mdy = float(flow[..., 1][moving].mean()) if moving.any() else 0.0
        stats.append(dict(time=round(start + i / fps, 3), mean_speed=float(mag.mean()),
                          pct_moving=100 * float(moving.mean()), mean_dx=mdx, mean_dy=mdy))
        if i % keep_every == 1 or keep_every == 1:
            k = VIEW_W / frame.shape[1]
            small = lambda im: cv2.cvtColor(cv2.resize(im, None, fx=k, fy=k), cv2.COLOR_BGR2RGB)
            panels.append((round(start + i / fps, 2), small(frame), small(flow_to_hsv(flow)),
                           small(draw_arrows(frame, flow, step=16))))
        prev_g = g
    cap.release()
    os.unlink(path)
    return dict(fps=fps, size=(prev.shape[1], prev.shape[0]), stats=pd.DataFrame(stats),
                panels=panels)


st.title("Optical flow")
st.write("Dense optical flow (Farnebäck) between every pair of consecutive frames, with the "
         "same settings as `optical_flow.py`. Colour shows the direction each pixel moves and "
         "brightness shows how fast. Compare a fixed camera with a moving one: in the first only "
         "the cars light up, in the second the whole roadside streams outward from the horizon.")

c1, c2, c3 = st.columns([1.1, 1.4, 1.2], gap="large")
with c1:
    source = st.radio("Video", ["Sample clip", "Upload your own"], horizontal=True)
if source == "Sample clip":
    with c2:
        name = st.selectbox("Sample", list(SAMPLES))
    with open(os.path.join(SAMPLE_DIR, SAMPLES[name]), "rb") as f:
        data = f.read()
    start = 0.0
    with c3:
        seconds = st.slider("Seconds to process", 1, 6, 4)
else:
    with c2:
        up = st.file_uploader("Short video with motion", type=["mp4", "mov", "avi", "m4v"])
    with c3:
        start = st.number_input("Start at (s)", 0.0, 600.0, 0.0, 0.5)
        seconds = st.slider("Seconds to process", 1, MAX_SECONDS, 4)
    if up is None:
        st.info(f"Upload a clip (20 MB max). The page processes up to {MAX_SECONDS} seconds, "
                f"scaled to {MAX_W} px wide.")
        st.stop()
    data = up.getvalue()

with st.spinner("Computing flow for every frame pair..."):
    res = process(data, float(start), float(seconds))
if res is None or not res["panels"]:
    st.error("Could not read frames from that video at the chosen start time.")
    st.stop()

df = res["stats"]
times = [p[0] for p in res["panels"]]
t = st.select_slider("Frame (seconds into the clip)", options=times, value=times[len(times) // 2])
_, frame, hsv, arrows = res["panels"][times.index(t)]
p1, p2, p3 = st.columns(3)
p1.image(frame, caption="Original")
p2.image(hsv, caption="Flow, colour coded")
p3.image(arrows, caption="Flow vectors (drawn 3× longer)")

k1, m1, m2, m3 = st.columns([0.8, 1, 1, 1])
k1.image(colour_wheel(110), caption="Colour key: hue = direction, brightness = speed", width=110)
m1.metric("Mean speed (px/frame)", f"{df.mean_speed.mean():.2f}")
m2.metric("Pixels moving (%)", f"{df.pct_moving.mean():.1f}",
          help="Share of pixels moving more than 1 px per frame, averaged over the clip")
m3.metric("Busiest frame (% moving)", f"{df.pct_moving.max():.1f}")
st.caption(f"Processed {len(df) + 1} frames at {res['size'][0]}×{res['size'][1]} px, "
           f"{res['fps']:.2f} fps. In the colour key the image y axis points down, so green "
           "means moving down the frame and purple means moving up.")

st.subheader("Motion over time")
st.write("A fixed camera keeps both curves low and flat: only the objects move. A moving camera "
         "raises both, because every pixel in the scene shifts.")
cc1, cc2 = st.columns(2)
with cc1:
    st.caption("Mean flow speed (px/frame)")
    st.line_chart(df.set_index("time")[["mean_speed"]], height=220)
with cc2:
    st.caption("Share of pixels moving more than 1 px/frame (%)")
    st.line_chart(df.set_index("time")[["pct_moving"]], height=220)
st.download_button("Download per-frame statistics (CSV)", df.to_csv(index=False),
                   file_name="flow_stats.csv", mime="text/csv")
