"""
README - sfm_book.py  (CSc 8830 Assignment 6, Part B: structure from motion, planar object)
==========================================================================================
What it does
    Reconstructs a flat book cover (14.5 cm x 20 cm) from four photos taken with an
    iPhone 17 Pro (24 mm-equivalent, 1932x2576 images) and recovers the four camera
    positions. Steps:
      1. Intrinsics K from the 35 mm-equivalent focal length.
      2. Cover corners in view 1 (overhead) by fitting lines to the four cover edges.
      3. SIFT matches on the cover between view 1 and each other view; RANSAC homography.
      4. Corners in views 2-4 by transferring view-1 corners through each homography.
      5. Homography decomposition H ~ K (R + t n^T / d) K^-1 -> R, t, n for each view.
      6. Linear (DLT) triangulation of the corners from all four views and of every
         SIFT match, then bundle adjustment (Levenberg-Marquardt) of all poses and
         points; scale fixed by the 14.5 cm width, 20 cm length kept as a check.
      7. Camera centres in a book frame (origin BL = bottom of spine, x along the
         bottom edge, y up the spine, z height above the cover).
      8. Independent check: focal length re-estimated per view from the known rectangle.

Outputs (in --out)
    results.txt              every number quoted in the report
    fig_matches.png          SIFT inliers per view pair
    fig_corners.png          corners in each view with reprojected 3D corners
    fig_3d.png               reconstructed cover, surface points and camera positions

Requirements
    pip install opencv-python numpy matplotlib scipy

Usage
    python sfm_book.py --images images --out results
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

VIEWS = ["view1_overhead", "view2_sideways", "view3_spine", "view4_front"]
W_CM, L_CM = 14.5, 20.0            # measured book width and length
F_EQ = 24.0                        # 35 mm-equivalent focal length (from photo info)
FULL_W, FULL_H = 3024, 4032        # native resolution of the iPhone sensor output
DIAG_35 = np.hypot(36, 24)         # 43.27 mm, diagonal of a 35 mm frame
NAMES = ["TL", "TR", "BR", "BL"]   # TL = top of spine; TL->TR = 14.5 cm, TR->BR = 20 cm


def intrinsics(w, h):
    """f_px = f_eq * image_diagonal_px / 43.27 mm, scaled to the actual image size."""
    f_full = F_EQ * np.hypot(FULL_W, FULL_H) / DIAG_35
    f = f_full * w / FULL_W
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), f_full


def view1_corners(img):
    """Fit a line to each edge of the cover (strongest colour edge in a band around
    a rough polygon from colour segmentation) and intersect neighbouring lines."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (((h >= 165) | (h <= 6)) & (s > 90) & (v > 30)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    mask = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    c = max(cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            key=cv2.contourArea)
    hull = cv2.convexHull(c)
    rough = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    rough = order_tl_tr_br_bl(rough.astype(float))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    ch = cv2.GaussianBlur(lab_img[..., 1], (0, 0), 1.5)          # a* channel: red vs not
    gx, gy = cv2.Sobel(ch, cv2.CV_64F, 1, 0), cv2.Sobel(ch, cv2.CV_64F, 0, 1)
    cen, lines, resid = rough.mean(0), [], []
    for i in range(4):
        a, b = rough[i], rough[(i + 1) % 4]
        L = np.linalg.norm(b - a); u = (b - a) / L; nrm = np.array([-u[1], u[0]])
        if (cen - a) @ nrm > 0:
            nrm = -nrm
        pts = []
        for t in np.linspace(0.1 * L, 0.9 * L, 300):
            offs = np.arange(-25, 26); xy = a + t * u + offs[:, None] * nrm
            xs, ys = xy[:, 0].round().astype(int), xy[:, 1].round().astype(int)
            prof = -(gx[ys, xs] * nrm[0] + gy[ys, xs] * nrm[1])
            k = int(np.argmax(prof))
            sub = 0.0
            if 0 < k < len(prof) - 1:
                d = prof[k - 1] - 2 * prof[k] + prof[k + 1]
                sub = 0.5 * (prof[k - 1] - prof[k + 1]) / d if d else 0.0
            pts.append(a + t * u + (offs[k] + sub) * nrm)
        pts = np.float32(pts)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        resid.append(float(np.median(np.abs((pts - [x0, y0]) @ np.array([-vy, vx])))))
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = np.array([intersect(lines[i - 1], lines[i]) for i in range(4)])
    return corners, mask, resid


def order_tl_tr_br_bl(p):
    s, d = p.sum(1), p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])


def intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    return p1 + t[0] * d1


def choose_decomposition(H, K, x1):
    """cv2 returns up to 4 (R, t, n). Keep those that place the matched points in
    front of camera 1 (n . K^-1 x > 0 for every point)."""
    _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    rays = (np.linalg.inv(K) @ np.c_[x1, np.ones(len(x1))].T).T
    keep = [(R, t.ravel(), n.ravel()) for R, t, n in zip(Rs, ts, ns)
            if np.all(rays @ n.ravel() > 0)]
    return keep


def triangulate(Ps, xs):
    """Linear DLT: each view adds x*(p3.X) - p1.X = 0 and y*(p3.X) - p2.X = 0."""
    A = []
    for P, (x, y) in zip(Ps, xs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3], np.array(A)


def project(P, X):
    x = P @ np.append(X, 1)
    return x[:2] / x[2]


def focal_from_rectangle(img_corners, cx, cy):
    """Zhang-style check: world rectangle (cm) -> image homography H = K[r1 r2 t].
    With omega = K^-T K^-1 = diag(1/f^2, 1/f^2, 1) after centring,
    r1.r2 = 0 and |r1| = |r2| each give an estimate of f."""
    world = np.float32([[0, 0], [W_CM, 0], [W_CM, L_CM], [0, L_CM]])
    img = np.float32(img_corners) - [cx, cy]
    H, _ = cv2.findHomography(world, img)
    h1, h2 = H[:, 0], H[:, 1]
    est = []
    num = h1[2] * h2[2]                      # from h1^T w h2 = 0
    den = h1[0] * h2[0] + h1[1] * h2[1]
    if num != 0 and -den / num > 0:
        est.append(np.sqrt(-den / num))
    num = h1[2] ** 2 - h2[2] ** 2            # from h1^T w h1 = h2^T w h2
    den = (h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)
    if num != 0 and den / num > 0:
        est.append(np.sqrt(den / num))
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = open(os.path.join(args.out, "results.txt"), "w")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

    imgs = [cv2.imread(os.path.join(args.images, v + ".jpeg")) for v in VIEWS]
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    h, w = gray[0].shape
    K, f_full = intrinsics(w, h)
    np.set_printoptions(precision=4, suppress=True)
    log(f"[K] f_full = 24 * {np.hypot(FULL_W, FULL_H):.1f} / {DIAG_35:.2f} = {f_full:.1f} px "
        f"at {FULL_W}x{FULL_H}; scaled to {w}x{h}: f = {K[0,0]:.1f} px")
    log("[K]\n" + str(K))

    # ---- corners in view 1
    C1, mask1, resid = view1_corners(imgs[0])
    log("[corners] view1", C1.round(2).tolist(), "line-fit median residuals px", np.round(resid, 2).tolist())

    # ---- SIFT + homographies view1 -> view j
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(gray[0], mask1 * 255)
    bf = cv2.BFMatcher()
    Hs, matches, corners = {}, {}, {0: C1}
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for j in (1, 2, 3):
        k2, d2 = sift.detectAndCompute(gray[j], None)
        ms = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
        p1 = np.float32([k1[a.queryIdx].pt for a in ms])
        p2 = np.float32([k2[a.trainIdx].pt for a in ms])
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
        inl = inl.ravel() == 1
        err = np.linalg.norm(cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
                             - p2[inl], axis=1)
        Hs[j] = H / H[2, 2]
        matches[j] = (p1[inl], p2[inl])
        corners[j] = cv2.perspectiveTransform(C1.reshape(-1, 1, 2), Hs[j]).reshape(-1, 2)
        log(f"[H] view1->{VIEWS[j]}: {len(ms)} ratio-test matches, {inl.sum()} RANSAC inliers, "
            f"transfer error median {np.median(err):.2f} px")
        log(f"[H]\n{Hs[j]}")
        log(f"[corners] {VIEWS[j]}", corners[j].round(2).tolist())
        both = np.hstack([cv2.resize(imgs[0], None, fx=0.25, fy=0.25),
                          cv2.resize(imgs[j], None, fx=0.25, fy=0.25)])
        axs[j - 1].imshow(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))
        for a, b in zip(p1[inl][::2], p2[inl][::2]):
            axs[j - 1].plot([a[0] / 4, b[0] / 4 + w / 4], [a[1] / 4, b[1] / 4], lw=0.6)
        axs[j - 1].set_title(f"view 1 <-> view {j+1}: {inl.sum()} inliers"); axs[j - 1].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_matches.png"), dpi=110); plt.close()

    # ---- decompose each homography; pick the normal consistent across the 3 views
    cands = {j: choose_decomposition(Hs[j], K, matches[j][0]) for j in (1, 2, 3)}
    best, best_spread = None, 1e9
    for a in cands[1]:
        for b in cands[2]:
            for c in cands[3]:
                ns = np.array([a[2], b[2], c[2]])
                spread = np.linalg.norm(ns - ns.mean(0), axis=1).sum()
                if spread < best_spread:
                    best, best_spread = (a, b, c), spread
    n = np.mean([s[2] for s in best], axis=0); n /= np.linalg.norm(n)
    log(f"[decomp] candidates kept per view: {[len(cands[j]) for j in (1,2,3)]}")
    for j, (R, t, nj) in zip((1, 2, 3), best):
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        log(f"[decomp] {VIEWS[j]}: n = {nj.round(4)}, t/d = {t.round(4)}, rotation angle {ang:.1f} deg")
        log(f"[decomp] R =\n{R}")
    log(f"[decomp] plane normal in camera-1 frame (mean) n = {n.round(4)}, "
        f"spread of the three normals {np.degrees(best_spread):.2f} deg-ish (sum of chord lengths)")

    # ---- projection matrices with d = 1, then triangulate the corners from 4 views
    Ps = [K @ np.hstack([np.eye(3), np.zeros((3, 1))])]
    Rts = [(np.eye(3), np.zeros(3))]
    for R, t, _ in best:
        Ps.append(K @ np.hstack([R, t.reshape(3, 1)]))
        Rts.append((R, t))
    X = []
    for k in range(4):
        Xk, A = triangulate(Ps, [corners[j][k] for j in range(4)])
        X.append(Xk)
        if k == 0:
            log(f"[triangulate] TL design matrix A (8x4), rows = x*p3 - p1, y*p3 - p2 per view:\n{A}")
    X = np.array(X)
    reproj = np.array([[np.linalg.norm(project(Ps[j], X[k]) - corners[j][k]) for k in range(4)]
                       for j in range(4)])
    log(f"[triangulate] corners (units of d):\n{X}")
    log(f"[triangulate] reprojection error px (rows views, cols TL TR BR BL):\n{reproj.round(2)}")

    # ---- bundle adjustment: refine poses of views 2-4 and all 3D points together.
    # Observations: 4 corners in 4 views + each SIFT match in view 1 and view j.
    # Camera 1 stays at K[I|0]; scale is fixed afterwards by the 14.5 cm width.
    obs, pts0 = [], [X[k] for k in range(4)]
    for k in range(4):
        for j in range(4):
            obs.append((j, k, corners[j][k]))
    for j in (1, 2, 3):
        for a, b in zip(*matches[j]):
            idx = len(pts0)
            pts0.append(triangulate([Ps[0], Ps[j]], [a, b])[0])
            obs.append((0, idx, a)); obs.append((j, idx, b))
    pts0 = np.array(pts0)
    cam_idx = np.array([o[0] for o in obs]); pt_idx = np.array([o[1] for o in obs])
    uv = np.array([o[2] for o in obs])
    x0 = np.concatenate([np.concatenate([cv2.Rodrigues(Rts[j][0])[0].ravel(), Rts[j][1]])
                         for j in (1, 2, 3)] + [pts0.ravel()])
    t2_norm = np.linalg.norm(Rts[1][1])

    def residuals(x):
        Rs = [np.eye(3)] + [cv2.Rodrigues(x[6*i:6*i+3])[0] for i in range(3)]
        ts = [np.zeros(3)] + [x[6*i+3:6*i+6] for i in range(3)]
        P3 = x[18:].reshape(-1, 3)
        r = np.empty((len(obs), 2))
        for j in range(4):
            sel = cam_idx == j
            Xc_ = (Rs[j] @ P3[pt_idx[sel]].T).T + ts[j]
            proj = (K @ Xc_.T).T
            r[sel] = proj[:, :2] / proj[:, 2:3] - uv[sel]
        gauge = 1e3 * (np.linalg.norm(x[3:6]) - t2_norm)     # pins the free scale
        return np.append(r.ravel(), gauge)

    r0 = residuals(x0)[:-1].reshape(-1, 2)
    sol = least_squares(residuals, x0, method="lm", max_nfev=2000)
    r1 = sol.fun[:-1].reshape(-1, 2)
    e0, e1 = np.linalg.norm(r0, axis=1), np.linalg.norm(r1, axis=1)
    log(f"[BA] {len(obs)} observations, {len(pts0)} points, {len(x0)} parameters")
    log(f"[BA] reprojection error before: mean {e0.mean():.2f} px, max {e0.max():.2f} px; "
        f"after: mean {e1.mean():.2f} px, max {e1.max():.2f} px")
    x = sol.x
    Rts = [(np.eye(3), np.zeros(3))] + [(cv2.Rodrigues(x[6*i:6*i+3])[0], x[6*i+3:6*i+6])
                                        for i in range(3)]
    Ps = [K @ np.hstack([R, t.reshape(3, 1)]) for R, t in Rts]
    Pall = x[18:].reshape(-1, 3)
    X = Pall[:4]
    reproj = np.array([[np.linalg.norm(project(Ps[j], X[k]) - corners[j][k]) for k in range(4)]
                       for j in range(4)])
    log(f"[BA] corners (units of camera-1 plane distance):\n{X}")
    log(f"[BA] corner reprojection error px (rows views, cols TL TR BR BL):\n{reproj.round(2)}")
    for j in (1, 2, 3):
        R, t = Rts[j]
        log(f"[BA] {VIEWS[j]}: R =\n{R}\n t = {t.round(4)}")

    # ---- metric scale from the width; length, diagonals, angles as checks
    width_u = (np.linalg.norm(X[1] - X[0]) + np.linalg.norm(X[2] - X[3])) / 2
    s = W_CM / width_u
    Xc = X * s
    sides = [np.linalg.norm(Xc[(i + 1) % 4] - Xc[i]) for i in range(4)]
    diag = [np.linalg.norm(Xc[2] - Xc[0]), np.linalg.norm(Xc[3] - Xc[1])]
    angs = []
    for i in range(4):
        a, b = Xc[i - 1] - Xc[i], Xc[(i + 1) % 4] - Xc[i]
        angs.append(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))
    cen = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - cen)
    plan = np.abs((Xc - cen) @ Vt[2])
    log(f"[metric] scale s = {W_CM} / {width_u:.5f} = {s:.3f} cm per unit; camera-1 to plane distance d = {s:.2f} cm")
    log(f"[metric] sides TL-TR, TR-BR, BR-BL, BL-TL (cm): {np.round(sides, 2).tolist()}")
    log(f"[metric] length error vs 20 cm: {np.mean([sides[1], sides[3]]) - L_CM:+.2f} cm "
        f"({100*(np.mean([sides[1], sides[3]]) / L_CM - 1):+.1f}%)")
    log(f"[metric] diagonals (cm): {np.round(diag, 2).tolist()}, expected {np.hypot(W_CM, L_CM):.2f}")
    log(f"[metric] corner angles (deg): {np.round(angs, 2).tolist()}")
    log(f"[metric] corner distance from best-fit plane (cm): {plan.round(3).tolist()}")

    # ---- dense surface points from SIFT matches (two-view triangulation)
    surf = Pall[4:] * s
    dsurf = (surf - cen) @ Vt[2]
    log(f"[surface] {len(surf)} SIFT points; distance from corner plane: median "
        f"{np.median(np.abs(dsurf)):.3f} cm, 90th pct {np.percentile(np.abs(dsurf), 90):.3f} cm")

    # ---- boundary: view-1 cover contour back-projected onto the plane
    cnt = max(cv2.findContours(mask1 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
              key=cv2.contourArea).reshape(-1, 2)[::8].astype(float)
    rays = (np.linalg.inv(K) @ np.c_[cnt, np.ones(len(cnt))].T).T
    nd = np.cross(X[1] - X[0], X[3] - X[0]); nd /= np.linalg.norm(nd)
    dd = nd @ X[0]
    boundary = rays * (dd / (rays @ nd))[:, None] * s

    # ---- book frame (right-handed): origin BL corner (bottom of spine), x along the
    # bottom edge towards BR (14.5 cm), y along the spine towards TL (20 cm), z = x cross y,
    # which points up out of the cover towards the cameras
    ex = (Xc[2] - Xc[3]); ex /= np.linalg.norm(ex)
    ey = (Xc[0] - Xc[3]); ey -= (ey @ ex) * ex; ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    Rb = np.vstack([ex, ey, ez])   # rows: book axes in camera-1 coords
    to_book = lambda P: (Rb @ (np.atleast_2d(P) - Xc[3]).T).T
    assert to_book(np.zeros(3))[0, 2] > 0, "camera 1 must sit above the cover"
    log("[book frame] corners (cm):", to_book(Xc).round(2).tolist())
    cams = []
    for j, (R, t) in enumerate(Rts):
        Cc = -R.T @ t * s                         # centre in camera-1 frame (cm)
        Cb = to_book(Cc)[0]
        view_dir = Rb @ (R.T @ np.array([0, 0, 1]))     # optical axis in book frame
        tilt = np.degrees(np.arccos(np.clip(-view_dir[2], -1, 1)))
        dist = np.linalg.norm(Cb - to_book(cen)[0])
        cams.append((Cb, view_dir, tilt, dist))
        log(f"[camera] {VIEWS[j]}: position (x, y, z) = {Cb.round(1).tolist()} cm, "
            f"distance to cover centre {dist:.1f} cm, tilt from straight down {tilt:.1f} deg")

    # ---- independent focal-length check from the known rectangle
    for j in range(4):
        est = focal_from_rectangle(corners[j], w / 2, h / 2)
        log(f"[f check] {VIEWS[j]}: f from rectangle = {np.round(est, 0).tolist()} px (K uses {K[0,0]:.0f})")

    # ---- figure: corners and reprojections
    fig, axs = plt.subplots(1, 4, figsize=(18, 6.5))
    for j in range(4):
        axs[j].imshow(cv2.cvtColor(imgs[j], cv2.COLOR_BGR2RGB))
        poly = np.vstack([corners[j], corners[j][:1]])
        axs[j].plot(poly[:, 0], poly[:, 1], "c-", lw=1.5)
        rp = np.array([project(Ps[j], X[k]) for k in range(4)])
        axs[j].plot(rp[:, 0], rp[:, 1], "r+", ms=14, mew=2)
        for k in range(4):
            axs[j].text(corners[j][k, 0], corners[j][k, 1], NAMES[k], color="yellow", fontsize=10)
        axs[j].set_title(f"view {j+1}: max reproj {reproj[j].max():.2f} px"); axs[j].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_corners.png"), dpi=110); plt.close()

    # ---- figure: 3D reconstruction in the book frame
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cb = to_book(Xc); sb = to_book(surf); bb = to_book(boundary)
    ax.scatter(sb[:, 0], sb[:, 1], sb[:, 2], s=2, c="gray", label="SIFT surface points")
    ax.plot(bb[:, 0], bb[:, 1], bb[:, 2], "g-", lw=1, label="boundary (view-1 contour)")
    cc = np.vstack([cb, cb[:1]])
    ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "r-o", lw=2, label="triangulated corners")
    for j, (Cb, vd, tilt, dist) in enumerate(cams):
        ax.scatter(*Cb, s=60, marker="^")
        ax.plot(*np.c_[Cb, Cb + 8 * vd], lw=1)
        ax.text(*Cb, f" cam {j+1}", fontsize=9)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
    ax.legend(loc="upper left", fontsize=8); ax.view_init(elev=22, azim=-60)
    ax.set_title("Cover and cameras, book frame")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(bb[:, 0], bb[:, 1], "g-", lw=1, label="boundary from view-1 contour")
    ax2.plot(cc[:, 0], cc[:, 1], "r-o", label="triangulated corners")
    ax2.plot([0, W_CM, W_CM, 0, 0], [0, 0, L_CM, L_CM, 0], "k--", lw=1, label="true 14.5 x 20 cm")
    ax2.scatter(sb[:, 0], sb[:, 1], s=2, c="gray")
    ax2.set_aspect("equal"); ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlabel("x (cm)"); ax2.set_ylabel("y (cm)"); ax2.set_title("Top view of the recovered plane")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_3d.png"), dpi=120); plt.close()
    out.close()


if __name__ == "__main__":
    main()
