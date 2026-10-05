"""
README - optical_flow.py  (CSc 8830 Assignment 6, Part A: optical flow visualization)
=====================================================================================
What it does
    Takes a 30-second motion segment from a video, computes dense optical flow
    (Farneback) between every pair of consecutive frames, and writes:
      1. <name>_flow.mp4   side-by-side video: original | HSV flow | arrow overlay
      2. <name>_stats.csv  per-frame flow statistics (mean/max magnitude, dominant
                           direction, % of pixels moving), used as evidence for
                           the "what can be inferred from optical flow" section.

HSV color code: hue = direction of motion, brightness = speed.
    red ~ right, green ~ down, cyan ~ left, purple/blue ~ up (OpenCV image coords).

Requirements
    pip install opencv-python numpy

Usage
    python optical_flow.py --video my_video.mp4 --start 5 --duration 30
    python optical_flow.py --video my_video.mp4 --start 0 --duration 30 --scale 0.5

Arguments
    --video     path to input video (required)
    --start     start time in seconds of the motion segment (default 0)
    --duration  length of segment in seconds (default 30)
    --scale     resize factor for speed, e.g. 0.5 halves resolution (default 1.0)
    --step      spacing in pixels between arrows in the arrow overlay (default 16)
    --outdir    output folder (default ./output)
"""

import argparse
import csv
import os

import cv2
import numpy as np


def flow_to_hsv(flow, max_mag=None):
    """Convert a flow field (H x W x 2) to a BGR image using the HSV color wheel."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # ang in radians
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # hue: 0-180 in OpenCV
    hsv[..., 1] = 255                                        # full saturation
    if max_mag is None:
        max_mag = np.percentile(mag, 99) + 1e-6              # robust normalization
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(frame, flow, step=16, min_mag=1.0):
    """Draw flow vectors on a sparse grid over the frame."""
    out = frame.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if np.hypot(dx, dy) >= min_mag:   # skip near-static pixels
                cv2.arrowedLine(out, (x, y), (int(x + dx * 3), int(y + dy * 3)),
                                (0, 255, 0), 1, tipLength=0.3)  # x3 for visibility
    return out


def label(img, text):
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)   # jump to segment start
    n_frames = int(args.duration * fps)

    ok, prev = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame of segment")
    if args.scale != 1.0:
        prev = cv2.resize(prev, None, fx=args.scale, fy=args.scale)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    writer = cv2.VideoWriter(os.path.join(args.outdir, f"{name}_flow.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, h))
    stats_f = open(os.path.join(args.outdir, f"{name}_stats.csv"), "w", newline="")
    stats = csv.writer(stats_f)
    stats.writerow(["frame", "time_s", "mean_mag", "max_mag",
                    "pct_moving", "mean_dx", "mean_dy", "dominant_dir_deg"])

    for i in range(1, n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farneback dense flow: pyramid scale 0.5, 3 levels, 15px window,
        # 3 iterations, poly_n=5, poly_sigma=1.2
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # --- statistics for the "what can we infer" analysis ---
        mag = np.hypot(flow[..., 0], flow[..., 1])
        moving = mag > 1.0                         # pixels moving > 1 px/frame
        if moving.any():
            mdx, mdy = flow[..., 0][moving].mean(), flow[..., 1][moving].mean()
        else:
            mdx = mdy = 0.0
        stats.writerow([i, round(args.start + i / fps, 3),
                        round(float(mag.mean()), 4), round(float(mag.max()), 4),
                        round(100 * moving.mean(), 2), round(float(mdx), 4),
                        round(float(mdy), 4),
                        round(float(np.degrees(np.arctan2(mdy, mdx))), 1)])

        # --- visualization ---
        panel = np.hstack([label(frame.copy(), "Original"),
                           label(flow_to_hsv(flow), "Flow (hue=dir, bright=speed)"),
                           label(draw_arrows(frame, flow, args.step), "Flow vectors")])
        writer.write(panel)

        prev_gray = gray
        if i % 100 == 0:
            print(f"processed {i}/{n_frames} frames")

    cap.release()
    writer.release()
    stats_f.close()
    print(f"Done. Outputs in {args.outdir}/")


if __name__ == "__main__":
    main()
