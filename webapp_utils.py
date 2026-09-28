"""Shared helpers for the Streamlit pages (image I/O, drawing, parsing)."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def add_path(*parts):
    """Make a module folder importable, e.g. add_path('module4', 'q1')."""
    p = os.path.join(ROOT, *parts)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def rgb(img):
    """OpenCV stores BGR; Streamlit shows RGB."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload(file, max_side=1280):
    """Decode an uploaded image and shrink it so the longest side is at most
    max_side. Returns the BGR image and the scale applied."""
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, 1.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def read_mask(file, shape):
    """Decode an uploaded binary mask and match it to the image size."""
    data = np.frombuffer(file.getvalue(), np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def parse_points(text):
    """'x1,y1; x2,y2' -> [(x1, y1), (x2, y2)]; ignores blanks and bad pairs."""
    pts = []
    for chunk in (text or "").replace("\n", ";").split(";"):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
            pts.append((int(parts[0]), int(parts[1])))
    return pts


def draw_prompt(img, rect, points=(), radius=0):
    """Preview of the user's box (yellow) and background clicks (red)."""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(2, int(round(max(img.shape[:2]) / 400)))
    x, y, w, h = rect
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 255), t)
    for (px, py) in points:
        if radius > 0:
            overlay = vis.copy()
            cv2.circle(overlay, (px, py), radius, (0, 0, 255), -1)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        cv2.circle(vis, (px, py), 3 * t, (0, 0, 255), -1)
    return vis


def overlay_mask(img, mask, colour=(0, 255, 0)):
    """Tint the mask area and outline it."""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tint = vis.copy()
    tint[mask > 0] = colour
    vis = cv2.addWeighted(tint, 0.3, vis, 0.7, 0)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    t = max(1, int(round(max(img.shape[:2]) / 250)))
    cv2.drawContours(vis, cs, -1, colour, t)
    return vis


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def metrics_rows(metrics):
    """Metric dict -> list of rows for st.dataframe."""
    names = {"IoU": "IoU", "Dice": "Dice", "Precision": "Precision",
             "Recall": "Recall", "Boundary_F(3px)": "Boundary F-score (3 px)"}
    return [{"Metric": names.get(k, k), "Value": round(float(v), 3)} for k, v in metrics.items()]
