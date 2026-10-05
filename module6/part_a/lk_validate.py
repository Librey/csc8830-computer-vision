"""
README - lk_validate.py  (CSc 8830 Assignment 6, Part A.4: validate tracking on two frames)
==========================================================================================
What it does
    Takes two consecutive frames from a video, picks Shi-Tomasi corners, and tracks
    each corner three independent ways:
      1. MY LK     : the Lucas-Kanade equations derived in A.2, coded from scratch in
                     NumPy (central differences, G d = e, Newton iterations, bilinear
                     interpolation from A.3, 3-level pyramid). No OpenCV tracking.
      2. OPENCV LK : cv2.calcOpticalFlowPyrLK, the reference implementation.
      3. MEASURED  : the actual location of the point in frame 2, found by matching a
                     21x21 patch from frame 1 against frame 2 (normalised
                     cross-correlation, exhaustive search, parabola fit for sub-pixel).
                     This uses no gradients and no LK assumptions.
    It reports the error of MY LK against the measured location and against OpenCV,
    saves zoomed crops for visual checking, and writes a full hand-worked example
    (every number of one LK solve) for the report.

Outputs (in --out)
    <name>_points.csv        per point: p, MY LK p', OpenCV p', measured p', errors
    <name>_crops.png         frame-1 patch vs frame-2 patch at each predicted location
    <name>_worked.txt        5x5-window hand calculation for the first point

Requirements
    pip install opencv-python numpy matplotlib

Usage
    python lk_validate.py --video Traffic_static_shot.mp4 --t 20 --moving_only
    python lk_validate.py --video Driving_pov.mp4 --t 25 --min_rows 0.42 --max_rows 0.72
"""

import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCALE = 0.5


# ---------------------------------------------------------------- A.3: bilinear
def bilinear(img, x, y):
    """I(x,y) = (1-a)(1-b)I00 + a(1-b)I10 + (1-a)b I01 + ab I11, vectorised.
    img is float32 (H, W); x, y are arrays of real coordinates."""
    H, W = img.shape
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    a, b = x - x0, y - y0
    I00 = img[y0, x0]
    I10 = img[y0, x0 + 1]
    I01 = img[y0 + 1, x0]
    I11 = img[y0 + 1, x0 + 1]
    return (1 - a) * (1 - b) * I00 + a * (1 - b) * I10 + (1 - a) * b * I01 + a * b * I11


# ---------------------------------------------------------------- A.2: LK
def gradients(img):
    """Central differences: Ix = (I(x+1)-I(x-1))/2, Iy likewise."""
    Ix = np.zeros_like(img)
    Iy = np.zeros_like(img)
    Ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    Iy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return Ix, Iy


def lk_level(I, J, Ix, Iy, p, d0, half, iters=20, eps=0.01):
    """Iterative LK at one pyramid level. p = point in I, d0 = initial displacement.
    Returns final d and the iteration history."""
    xs = np.arange(-half, half + 1)
    gx, gy = np.meshgrid(xs, xs)
    wx, wy = p[0] + gx.ravel(), p[1] + gy.ravel()       # window pixels in I
    Iw = bilinear(I, wx, wy)
    ix = bilinear(Ix, wx, wy)
    iy = bilinear(Iy, wx, wy)
    G = np.array([[np.sum(ix * ix), np.sum(ix * iy)],
                  [np.sum(ix * iy), np.sum(iy * iy)]])
    d = np.array(d0, float)
    hist = []
    for k in range(iters):
        Jw = bilinear(J, wx + d[0], wy + d[1])            # J(x + d_k), A.3
        It = Jw - Iw                                      # I_t,k
        e = -np.array([np.sum(ix * It), np.sum(iy * It)])
        dd = np.linalg.solve(G, e)
        d = d + dd
        hist.append((k + 1, d[0], d[1], np.hypot(*dd)))
        if np.hypot(*dd) < eps:
            break
    return d, hist, G


def my_lk(I, J, p, half=7, levels=3):
    """Pyramidal LK: coarse to fine, d doubles at each finer level."""
    pyrI, pyrJ = [I], [J]
    for _ in range(levels - 1):
        pyrI.append(cv2.pyrDown(pyrI[-1]))
        pyrJ.append(cv2.pyrDown(pyrJ[-1]))
    d = np.zeros(2)
    for L in range(levels - 1, -1, -1):
        Ix, Iy = gradients(pyrI[L])
        pL = np.array(p) / (2 ** L)
        d, hist, G = lk_level(pyrI[L], pyrJ[L], Ix, Iy, pL, d, half)
        if L > 0:
            d = d * 2
    return d, hist, G


# ---------------------------------------------------------------- ground truth
def measured_location(I, J, p, half=10, search=20):
    """Actual location of p in J: exhaustive NCC match of a (2h+1)^2 patch,
    then a 1D parabola fit in x and y for sub-pixel precision."""
    x, y = int(round(p[0])), int(round(p[1]))
    tpl = I[y - half:y + half + 1, x - half:x + half + 1]
    y0, x0 = y - half - search, x - half - search
    region = J[y0:y + half + search + 1, x0:x + half + search + 1]
    if tpl.shape != (2 * half + 1,) * 2 or region.shape[0] < tpl.shape[0] + 2 * search:
        return None, 0.0
    R = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(R)
    def peak(m1, c, p1):
        den = m1 - 2 * c + p1
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den
    sx = peak(R[my, mx - 1], R[my, mx], R[my, mx + 1]) if 0 < mx < R.shape[1] - 1 else 0
    sy = peak(R[my - 1, mx], R[my, mx], R[my + 1, mx]) if 0 < my < R.shape[0] - 1 else 0
    # patch top-left (x0+mx, y0+my) -> centre
    return np.array([x0 + mx + sx + half + (p[0] - x),
                     y0 + my + sy + half + (p[1] - y)]), score


# ---------------------------------------------------------------- worked example
def worked_example(I, J, p, f):
    """Single-level, 5x5 window, one LK solve written out number by number."""
    x, y = int(round(p[0])), int(round(p[1]))
    Ix, Iy = gradients(I)
    sl = (slice(y - 2, y + 3), slice(x - 2, x + 3))
    ix, iy = Ix[sl], Iy[sl]
    it = J[sl] - I[sl]
    S = lambda m: float(np.sum(m))
    sxx, sxy, syy = S(ix * ix), S(ix * iy), S(iy * iy)
    sxt, syt = S(ix * it), S(iy * it)
    det = sxx * syy - sxy ** 2
    u = (-syy * sxt + sxy * syt) / det
    v = (sxy * sxt - sxx * syt) / det
    lam = np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]]))
    np.set_printoptions(precision=1, suppress=True, linewidth=120)
    f.write(f"Point p = ({x}, {y}), 5x5 window, first iteration (d0 = 0)\n\n")
    for name, M in [("I (frame 1)", I[sl]), ("J (frame 2)", J[sl]),
                    ("Ix", ix), ("Iy", iy), ("It = J - I", it)]:
        f.write(f"{name}:\n{M}\n\n")
    f.write(f"sum Ix^2 = {sxx:.1f}\nsum IxIy = {sxy:.1f}\nsum Iy^2 = {syy:.1f}\n"
            f"sum IxIt = {sxt:.1f}\nsum IyIt = {syt:.1f}\n\n"
            f"G = [[{sxx:.1f}, {sxy:.1f}], [{sxy:.1f}, {syy:.1f}]]\n"
            f"e = [{-sxt:.1f}, {-syt:.1f}]\n"
            f"det G = {det:.1f}\n"
            f"eigenvalues of G = {lam[0]:.1f}, {lam[1]:.1f}\n\n"
            f"u = (-sumIy2*sumIxIt + sumIxIy*sumIyIt)/det = {u:.4f}\n"
            f"v = (sumIxIy*sumIxIt - sumIx2*sumIyIt)/det = {v:.4f}\n")
    return dict(p=(x, y), sxx=sxx, sxy=sxy, syy=syy, sxt=sxt, syt=syt,
                det=det, u=u, v=v, lam=lam)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, required=True, help="time of frame 1 (s)")
    ap.add_argument("--n", type=int, default=6, help="points to track")
    ap.add_argument("--moving_only", action="store_true",
                    help="only pick corners on moving objects (fixed-camera videos)")
    ap.add_argument("--min_rows", type=float, default=0.0,
                    help="ignore corners above this fraction of image height")
    ap.add_argument("--max_rows", type=float, default=1.0,
                    help="ignore corners below this fraction of image height")
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000)
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    _, A = cap.read()
    _, B = cap.read()
    cap.release()
    A = cv2.resize(A, None, fx=SCALE, fy=SCALE)
    B = cv2.resize(B, None, fx=SCALE, fy=SCALE)
    I = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    J = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = I.shape

    # candidate corners (Shi-Tomasi), away from borders
    mask = np.zeros((H, W), np.uint8)
    mask[max(40, int(args.min_rows * H)):int(args.max_rows * H) - 40, 40:W - 40] = 255
    if args.moving_only:
        flow = cv2.calcOpticalFlowFarneback(I.astype(np.uint8), J.astype(np.uint8),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        moving = (np.linalg.norm(flow, axis=2) > 1.0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, moving)
    pts = cv2.goodFeaturesToTrack(I.astype(np.uint8), args.n, 0.05, 25, mask=mask,
                                  blockSize=7)
    pts = pts.reshape(-1, 2)

    # OpenCV reference
    cv_p, st, _ = cv2.calcOpticalFlowPyrLK(I.astype(np.uint8), J.astype(np.uint8),
                                           pts.astype(np.float32), None,
                                           winSize=(15, 15), maxLevel=2)

    rows = []
    for i, p in enumerate(pts):
        d, hist, G = my_lk(I, J, p)
        mine = p + d
        meas, score = measured_location(I, J, p)
        if meas is None:
            continue
        lam = np.linalg.eigvalsh(G)
        rows.append(dict(id=len(rows) + 1, x=p[0], y=p[1],
                         my_x=mine[0], my_y=mine[1], my_dx=d[0], my_dy=d[1],
                         iters=len(hist), lam2=lam[0],
                         cv_x=cv_p[i, 0], cv_y=cv_p[i, 1],
                         meas_x=meas[0], meas_y=meas[1], ncc=score,
                         err_meas=float(np.hypot(*(mine - meas))),
                         err_cv=float(np.hypot(*(mine - cv_p[i])))))

    with open(f"{args.out}/{name}_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    # crops: frame 1 patch at p, frame 2 patch at my predicted p'
    n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5))
    hw = 18
    for k, r in enumerate(rows):
        for row, (img, cx, cy, lab) in enumerate([(A, r["x"], r["y"], "frame t"),
                                                  (B, r["my_x"], r["my_y"], "frame t+1")]):
            M = np.float32([[1, 0, hw - cx], [0, 1, hw - cy]])
            crop = cv2.warpAffine(img, M, (2 * hw + 1, 2 * hw + 1),
                                  flags=cv2.INTER_LINEAR)
            ax[row, k].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                              interpolation="nearest")
            ax[row, k].axhline(hw, color="c", lw=0.6)
            ax[row, k].axvline(hw, color="c", lw=0.6)
            ax[row, k].set_xticks([]); ax[row, k].set_yticks([])
            if row == 0:
                ax[row, k].set_title(f"pt {r['id']}: p=({r['x']:.0f},{r['y']:.0f})",
                                     fontsize=8)
            else:
                ax[row, k].set_xlabel(f"p'=({r['my_x']:.1f},{r['my_y']:.1f})\n"
                                      f"err {r['err_meas']:.2f} px", fontsize=8)
        ax[0, 0].set_ylabel("frame t, centred on p")
        ax[1, 0].set_ylabel("frame t+1, centred on p'")
    plt.tight_layout()
    plt.savefig(f"{args.out}/{name}_crops.png", dpi=140)
    plt.close()

    # worked example on the point with the smallest motion (single level is valid)
    best = min(rows, key=lambda r: np.hypot(r["my_dx"], r["my_dy"]))
    with open(f"{args.out}/{name}_worked.txt", "w") as f:
        f.write(f"{name}: frame {idx} -> {idx + 1} (t = {args.t} s), image {W}x{H}\n\n")
        wk = worked_example(I, J, (best["x"], best["y"]), f)
        f.write(f"\nFull pyramidal LK for this point: d = ({best['my_dx']:.3f}, "
                f"{best['my_dy']:.3f}), measured d = ({best['meas_x']-best['x']:.3f}, "
                f"{best['meas_y']-best['y']:.3f})\n")

    print(f"{name}: frame {idx}->{idx + 1}, {len(rows)} points")
    for r in rows:
        print(f"  pt{r['id']} p=({r['x']:.1f},{r['y']:.1f}) d=({r['my_dx']:.2f},{r['my_dy']:.2f}) "
              f"iters={r['iters']} lam2={r['lam2']:.0f} | err vs measured {r['err_meas']:.3f}px "
              f"(ncc {r['ncc']:.3f}) | vs OpenCV {r['err_cv']:.3f}px")
    print(f"  mean err vs measured {np.mean([r['err_meas'] for r in rows]):.3f} px, "
          f"vs OpenCV {np.mean([r['err_cv'] for r in rows]):.3f} px")
    print(f"  worked example point {wk['p']}: one-step 5x5 (u,v)=({wk['u']:.3f},{wk['v']:.3f})")


if __name__ == "__main__":
    main()
