"""
README - flow_analysis.py  (CSc 8830 Assignment 6, Part A: what optical flow tells us)
======================================================================================
What it does
    Produces the evidence figures for the "what can be inferred from optical flow"
    section of the report. Run optical_flow.py first: this script reads its
    <name>_stats.csv files and recomputes Farneback flow on a few chosen frames.

    fig1_segmentation.png   traffic: thresholded flow -> moving-car mask and blob count
    fig2_directions.png     traffic: flow direction histograms for the two carriageways
    fig3_foe.png            driving: focus of expansion estimated from the flow field
    fig4_parallax.png       driving: flow speed by image region (far vs near)
    fig5_timeseries.png     both: mean speed and % moving pixels over the 30 s
    fig6_failures.png       driving: plain asphalt vs lane dashes, where flow fails
    analysis_numbers.txt    every number quoted in the report

Requirements
    pip install opencv-python numpy matplotlib pandas

Usage
    python flow_analysis.py --traffic Traffic_static_shot.mp4 --driving Driving_pov.mp4 \
        --stats output --out figures
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCALE = 0.5          # same resolution as the flow videos
MOVE_THR = 1.0       # px/frame: a pixel counts as moving above this


def frame_pair(path, t):
    """Return two consecutive frames (BGR, scaled) starting at time t seconds."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok1, a = cap.read()
    ok2, b = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise SystemExit(f"cannot read frames at {t}s from {path}")
    a = cv2.resize(a, None, fx=SCALE, fy=SCALE)
    b = cv2.resize(b, None, fx=SCALE, fy=SCALE)
    return a, b


def farneback(a, b):
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def flow_hsv(flow, vmax=None):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    vmax = vmax or np.percentile(mag, 99) + 1e-6
    hsv[..., 2] = np.clip(mag / vmax * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def estimate_foe(flow, mask):
    """Least-squares focus of expansion.
    Each moving pixel p with flow d defines a line through p along d. With normal
    n = (-dy, dx)/|d|, a point x on that line satisfies n.x = n.p. Stack all rows,
    weight by |d|, and solve for the x closest to every line."""
    ys, xs = np.nonzero(mask)
    d = flow[ys, xs]
    mag = np.linalg.norm(d, axis=1)
    n = np.stack([-d[:, 1], d[:, 0]], 1) / mag[:, None]
    p = np.stack([xs, ys], 1).astype(float)
    w = np.sqrt(mag)[:, None]
    A, b = n * w, (np.sum(n * p, 1) * w[:, 0])
    foe, *_ = np.linalg.lstsq(A, b, rcond=None)
    # check: fraction of vectors pointing away from the FOE (expansion)
    away = np.sum((p - foe) * d, 1) > 0
    return foe, away.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", required=True)
    ap.add_argument("--driving", required=True)
    ap.add_argument("--stats", default="output")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--t_traffic", type=float, default=20.0)
    ap.add_argument("--t_driving", type=float, default=25.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    notes = []

    # ---------- traffic: segmentation (fig 1) ----------
    a, b = frame_pair(args.traffic, args.t_traffic)
    flow = farneback(a, b)
    mag = np.linalg.norm(flow, axis=2)
    mask = (mag > MOVE_THR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 40]
    overlay = rgb(a).copy()
    for x, y, w, h, _ in blobs:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
    ax[0].imshow(rgb(a)); ax[0].set_title(f"Traffic frame, t = {args.t_traffic:.0f} s")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title(f"Pixels moving > {MOVE_THR} px/frame")
    ax[2].imshow(overlay); ax[2].set_title(f"{len(blobs)} moving regions found")
    for x in ax: x.axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig1_segmentation.png", dpi=130); plt.close()
    notes.append(f"[fig1] traffic t={args.t_traffic}s: moving pixels {100*mask.mean():.2f}%, "
                 f"moving regions (area>=40px) {len(blobs)}")

    # ---------- traffic: two carriageways (fig 2) ----------
    H, W = mag.shape
    moving = mag > MOVE_THR
    left = moving.copy(); left[:, W // 2:] = False
    right = moving.copy(); right[:, :W // 2] = False
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].imshow(flow_hsv(flow)); ax[0].axvline(W // 2, color="w", ls="--")
    ax[0].set_title("Flow (hue = direction)"); ax[0].axis("off")
    for k, (m, name, col) in enumerate([(left, "left half", "tab:green"),
                                        (right, "right half", "tab:purple")]):
        ang = np.degrees(np.arctan2(flow[..., 1][m], flow[..., 0][m]))
        sp = mag[m]
        ax[k + 1].hist(ang, bins=36, range=(-180, 180), color=col)
        ax[k + 1].set_xlabel("flow direction (deg; 0 = right, 90 = down, -90 = up)")
        ax[k + 1].set_title(f"{name}: median {np.median(ang):.0f} deg, "
                            f"mean speed {sp.mean():.2f} px/frame")
        notes.append(f"[fig2] traffic {name}: median direction {np.median(ang):.1f} deg, "
                     f"mean speed {sp.mean():.2f} px/frame, n={m.sum()}")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig2_directions.png", dpi=130); plt.close()

    # ---------- driving: focus of expansion (fig 3) ----------
    a, b = frame_pair(args.driving, args.t_driving)
    flow = farneback(a, b)
    mag = np.linalg.norm(flow, axis=2)
    H, W = mag.shape
    dash_row = int(0.68 * H)                 # dashboard starts here; exclude it
    m = mag > MOVE_THR
    m[dash_row:] = False
    foe, frac_away = estimate_foe(flow, m)
    vis = rgb(a).copy()
    step = 14
    for y in range(step // 2, dash_row, step):
        for x in range(step // 2, W, step):
            if m[y, x]:
                dx, dy = flow[y, x]
                cv2.arrowedLine(vis, (x, y), (int(x + 3 * dx), int(y + 3 * dy)),
                                (0, 255, 0), 1, tipLength=0.3)
    cv2.drawMarker(vis, (int(foe[0]), int(foe[1])), (255, 0, 0),
                   cv2.MARKER_CROSS, 30, 3)
    # FOE over many frame pairs, to show it stays put
    foes = []
    for t in np.arange(args.t_driving - 14, args.t_driving + 14, 2.0):
        fa, fb = frame_pair(args.driving, t)
        ff = farneback(fa, fb)
        mm = np.linalg.norm(ff, axis=2) > MOVE_THR
        mm[dash_row:] = False
        foes.append(estimate_foe(ff, mm)[0])
    foes = np.array(foes)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].imshow(vis); ax[0].axhline(dash_row, color="y", ls=":")
    ax[0].set_title(f"Flow vectors (x3) and estimated FOE (red +) at "
                    f"({foe[0]:.0f}, {foe[1]:.0f})"); ax[0].axis("off")
    ax[1].imshow(rgb(a)); ax[1].scatter(foes[:, 0], foes[:, 1], c="r", s=18)
    ax[1].set_title(f"FOE in {len(foes)} frame pairs over 28 s"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig3_foe.png", dpi=130); plt.close()
    notes.append(f"[fig3] driving t={args.t_driving}s: image {W}x{H}, FOE=({foe[0]:.1f},{foe[1]:.1f}), "
                 f"{100*frac_away:.1f}% of vectors point away from FOE; FOE over "
                 f"{len(foes)} pairs: mean ({foes[:,0].mean():.1f},{foes[:,1].mean():.1f}), "
                 f"std ({foes[:,0].std():.1f},{foes[:,1].std():.1f})")

    # ---------- driving: parallax by region (fig 4) ----------
    regions = {"sky": (0, int(0.25 * H), 0, W),
               "far mountains": (int(0.25 * H), int(0.42 * H), 0, W),
               "near roadside (left)": (int(0.45 * H), dash_row, 0, int(0.25 * W)),
               "near roadside (right)": (int(0.45 * H), dash_row, int(0.75 * W), W),
               "dashboard": (dash_row + 10, H, 0, W)}
    # average over the same 14 frame pairs for stability
    acc = {k: [] for k in regions}
    for t in np.arange(args.t_driving - 14, args.t_driving + 14, 2.0):
        fa, fb = frame_pair(args.driving, t)
        mm = np.linalg.norm(farneback(fa, fb), axis=2)
        for k, (y0, y1, x0, x1) in regions.items():
            acc[k].append(mm[y0:y1, x0:x1].mean())
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    show = rgb(a).copy()
    for k, (y0, y1, x0, x1) in regions.items():
        cv2.rectangle(show, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (255, 255, 0), 2)
        cv2.putText(show, k, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)
    ax[0].imshow(show); ax[0].axis("off"); ax[0].set_title("Regions")
    ax[1].barh(list(means), list(means.values()), color="tab:orange")
    ax[1].set_xlabel("mean flow speed (px/frame), 14 frame pairs")
    ax[1].invert_yaxis()
    plt.tight_layout(); plt.savefig(f"{args.out}/fig4_parallax.png", dpi=130); plt.close()
    for k, v in means.items():
        notes.append(f"[fig4] driving region '{k}': mean speed {v:.3f} px/frame")

    # ---------- both: time series (fig 5) ----------
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=False)
    for name, col in [("Traffic_static_shot", "tab:blue"), ("Driving_pov", "tab:red")]:
        d = pd.read_csv(os.path.join(args.stats, f"{name}_stats.csv"))
        t = d["time_s"] - d["time_s"].iloc[0]
        ax[0].plot(t, d["mean_mag"], color=col, lw=1, label=name)
        ax[1].plot(t, d["pct_moving"], color=col, lw=1, label=name)
        notes.append(f"[fig5] {name}: mean speed {d.mean_mag.mean():.3f} "
                     f"(min {d.mean_mag.min():.3f}, max {d.mean_mag.max():.3f}) px/frame; "
                     f"moving pixels {d.pct_moving.mean():.2f}% "
                     f"(min {d.pct_moving.min():.2f}, max {d.pct_moving.max():.2f})")
    ax[0].set_ylabel("mean speed (px/frame)"); ax[0].legend()
    ax[1].set_ylabel("% pixels moving"); ax[1].set_xlabel("seconds into 30 s segment")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig5_timeseries.png", dpi=130); plt.close()

    # ---------- driving: where flow fails (fig 6) ----------
    # Lane dashes and the asphalt around them sit in the same narrow strip and the
    # same rows, so they share the same true motion and distance from the FOE.
    # Any difference in measured flow comes from texture alone.
    y0, y1 = int(0.56 * H), int(0.66 * H)
    res = []
    for t in np.arange(args.t_driving - 14, args.t_driving + 14, 2.0):
        fa, fb = frame_pair(args.driving, t)
        mm = np.linalg.norm(farneback(fa, fb), axis=2)
        g = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY).astype(float)
        xs0, xs1 = int(0.40 * W), int(0.58 * W)
        xc = xs0 + int(np.argmax(g[y0:y1, xs0:xs1].max(0)))   # centre dash column
        G, M = g[y0:y1, xc - 12:xc + 13], mm[y0:y1, xc - 12:xc + 13]
        dash, asph = G > np.percentile(G, 90), G < np.percentile(G, 50)
        res.append((M[dash].mean(), M[asph].mean()))
    res = np.array(res)
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    tex = cv2.cornerMinEigenVal(gray, 7, 3)    # min eigenvalue of A^T A per pixel
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    ax[0].imshow(np.log1p(tex * 1e4)[:dash_row], cmap="magma"); ax[0].axis("off")
    ax[0].set_title("Texture: min eigenvalue of A^T A (bright = trackable)")
    ax[1].imshow(flow_hsv(flow)[:dash_row]); ax[1].axis("off")
    ax[1].set_title("Flow: sky and plain asphalt stay dark")
    k = np.arange(len(res))
    ax[2].bar(k - 0.2, res[:, 0], 0.4, label="white lane dash")
    ax[2].bar(k + 0.2, res[:, 1], 0.4, label="asphalt, same strip")
    ax[2].set_xlabel("frame pair (every 2 s)"); ax[2].set_ylabel("flow speed (px/frame)")
    ax[2].set_title("Same true motion, different measured flow"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{args.out}/fig6_failures.png", dpi=130); plt.close()
    notes.append(f"[fig6] centre strip over {len(res)} pairs: dash {res[:,0].mean():.2f} px/frame, "
                 f"asphalt {res[:,1].mean():.2f} px/frame; asphalt lower in "
                 f"{int((res[:,1] < res[:,0]).sum())}/{len(res)} pairs; sky mean speed "
                 f"{means['sky']:.3f}")

    with open(f"{args.out}/analysis_numbers.txt", "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
