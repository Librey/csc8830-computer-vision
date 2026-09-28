"""
===============================================================================
CSc 8830 Computer Vision, Module 4, Question 1
Human boundary extraction from an RGB image using classical OpenCV methods
(no deep learning or machine learning)
Author: Liberty Ikpeogu
===============================================================================

README
------
Install:
    pip install opencv-python numpy

Run:
    # Draw a box around the person with the mouse, then press ENTER or SPACE
    python q1_rgb_human_boundary.py --image person.jpg

    # Pass the box as x,y,w,h instead
    python q1_rgb_human_boundary.py --image person.jpg --rect 120,40,300,620

    # Compare against a SAM2 mask (binary PNG, person in white)
    python q1_rgb_human_boundary.py --image person.jpg --rect 120,40,300,620 \
        --sam_mask person_sam2.png

    # Mark areas that are not the person (GrabCut only), like SAM2's Remove
    python q1_rgb_human_boundary.py --image zidane.jpg --rect 115,195,1045,525 \
        --bg_points "860,300;850,520;880,680;950,620;1000,400;1050,500;1060,650;820,420"

    # Messi photo: clicks on the grass and ball, with smaller discs
    python q1_rgb_human_boundary.py --image messi5.jpg --rect 65,55,395,287 \
        --bg_points "110,300;170,320;360,310" --bg_radius 25

    # Switch to the watershed method
    python q1_rgb_human_boundary.py --image person.jpg --method watershed

The script writes to --out_dir (default ./results_q1):
    <name>_mask.png        binary mask, 255 marks the person
    <name>_boundary.png    input image with the boundary drawn in green
    <name>_compare.png     our mask, the SAM2 mask, and a difference map
                           (written only when you pass --sam_mask)
It prints IoU, Dice, precision, recall and boundary F-score to the console.

Methods
-------
grabcut (default)
    cv2.grabCut takes the user's box, models foreground and background
    colour, and solves a min-cut on the pixel graph. The script then cleans
    the mask with morphology, keeps the largest connected component, fills
    holes, and traces the boundary with cv2.findContours.
    GrabCut fits Gaussian mixtures to the colours inside and outside the box.
    It trains nothing ahead of time. If the course rules treat GMMs as ML,
    use --method watershed.

watershed
    The script builds colour histograms of the pixels inside and outside the
    box. Each pixel scores h_in / (h_in + h_out) for its colour, so colours
    found mostly inside the box score high. Otsu's threshold inside the box gives a rough person
    mask. The eroded rough mask seeds the person; pixels outside the box, or
    far from the rough mask, seed the background. cv2.watershed floods from
    both seeds, using colour differences between neighbouring pixels, and
    draws the boundary where the two floods meet.
===============================================================================
"""

import argparse
import os
import sys

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def resize_max(img, max_side=1024):
    """Shrink the image so its longest side is at most max_side, which keeps
    GrabCut fast. Returns the resized image and the scale factor."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def largest_component(mask, keep_ratio=0.15):
    """Keep the largest foreground blob plus any blob at least `keep_ratio`
    of its size. Accessories such as a collar ring can split the head from
    the torso, so keeping only one blob would throw the head away."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]  # label 0 is the background
    keep = 1 + np.flatnonzero(areas >= keep_ratio * areas.max())
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


def fill_holes(mask):
    """Fill interior holes by flood-filling the background from the border."""
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def clean_mask(mask, k=5):
    """Open to remove specks, close to bridge small gaps, keep the largest
    blob, and fill holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = largest_component(mask)
    mask = fill_holes(mask)
    return mask


def get_boundary(mask):
    """Return the outer contours of the mask with every boundary pixel kept
    (CHAIN_APPROX_NONE)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# -----------------------------------------------------------------------------
# Method 1: GrabCut
# -----------------------------------------------------------------------------
def segment_grabcut(img, rect, iters=6, bg_points=None, radius=45):
    """Segment the person inside `rect` (x, y, w, h) with GrabCut.
    `bg_points` is an optional list of (x, y) clicks on things that are NOT
    the person, such as a second person inside the box. Each click paints a
    disc of definite background, the same idea as SAM2's "Remove" click."""
    gc_mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)  # OpenCV needs these buffers for the colour models
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    if bg_points:
        for (px, py) in bg_points:
            cv2.circle(gc_mask, (px, py), radius, cv2.GC_BGD, -1)
        cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
    # GC_FGD (1) and GC_PR_FGD (3) are foreground labels
    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return clean_mask(mask)


# -----------------------------------------------------------------------------
# Method 2: Marker-based watershed (fully classical)
# -----------------------------------------------------------------------------
def segment_watershed(img, rect):
    """Segment the person inside `rect` with marker-controlled watershed."""
    x, y, w, h = rect
    H, W = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Colour model: quantise HSV into 30 x 32 x 8 bins and build one histogram
    # for pixels inside the box and one for pixels outside. For each colour,
    # score = h_in / (h_in + h_out). Colours that show up mostly inside the
    # box (the person) score near 255; colours common outside score low.
    hb = (hsv[..., 0].astype(np.int32) * 30) // 180
    sb = (hsv[..., 1].astype(np.int32) * 32) // 256
    vb = (hsv[..., 2].astype(np.int32) * 8) // 256
    idx = (hb * 32 + sb) * 8 + vb
    outside = np.ones((H, W), np.uint8)
    outside[y:y + h, x:x + w] = 0
    out_b = outside > 0
    n_bins = 30 * 32 * 8
    h_in = np.bincount(idx[~out_b], minlength=n_bins) / max((~out_b).sum(), 1)
    h_out = np.bincount(idx[out_b], minlength=n_bins) / max(out_b.sum(), 1)
    ratio = h_in / (h_in + h_out + 1e-9)
    fg_score = cv2.GaussianBlur((ratio[idx] * 255).astype(np.uint8), (9, 9), 0)

    # Otsu inside the box splits "person-like" from "background-like" pixels
    _, roi_fg = cv2.threshold(fg_score[y:y + h, x:x + w], 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rough = np.zeros((H, W), np.uint8)
    rough[y:y + h, x:x + w] = roi_fg
    rough = clean_mask(rough, k=7)

    # Seeds. Sure person: the rough mask eroded. Sure background: outside the
    # box, plus pixels inside the box far from the rough mask. Watershed
    # decides the band in between, where the true boundary lies.
    sure_fg = cv2.erode(rough, np.ones((11, 11), np.uint8))
    if sure_fg.sum() == 0:  # if erosion wiped out the seed, use a central ellipse
        cv2.ellipse(sure_fg, (x + w // 2, y + h // 2), (w // 6, h // 3), 0, 0, 360, 255, -1)
    near = cv2.dilate(rough, np.ones((25, 25), np.uint8))
    markers = np.zeros((H, W), np.int32)
    markers[(outside > 0) | (near == 0)] = 1  # label 1 = background
    markers[sure_fg > 0] = 2                  # label 2 = person
    # watershed assigns the remaining 0 pixels

    cv2.watershed(blur, markers)  # boundary pixels become -1
    mask = np.where(markers == 2, 255, 0).astype(np.uint8)
    return clean_mask(mask)


# -----------------------------------------------------------------------------
# Evaluation against SAM2
# -----------------------------------------------------------------------------
def boundary_f_score(pred, gt, tol=3):
    """Boundary F-score. Precision counts our boundary pixels that fall within
    `tol` px of the SAM2 boundary; recall counts the reverse."""
    def edge(m):
        return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

    pe, ge = edge(pred), edge(gt)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    ge_d = cv2.dilate(ge.astype(np.uint8), k) > 0
    pe_d = cv2.dilate(pe.astype(np.uint8), k) > 0
    prec = (pe & ge_d).sum() / max(pe.sum(), 1)
    rec = (ge & pe_d).sum() / max(ge.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def compare_masks(pred, gt):
    """Compute region and boundary metrics between our mask and the SAM2 mask."""
    p, g = pred > 127, gt > 127
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return {
        "IoU": tp / max(tp + fp + fn, 1),
        "Dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "Precision": tp / max(tp + fp, 1),
        "Recall": tp / max(tp + fn, 1),
        "Boundary_F(3px)": boundary_f_score(pred, gt, tol=3),
    }


def make_comparison_image(img, pred, gt):
    """Build three panels: our boundary, the SAM2 boundary, and a difference
    map (green for both, red for ours only, blue for SAM2 only)."""
    def overlay(mask, color):
        out = img.copy()
        for c in get_boundary(mask):
            cv2.drawContours(out, [c], -1, color, 2)
        return out

    diff = np.zeros_like(img)
    p, g = pred > 127, gt > 127
    diff[p & g] = (0, 200, 0)
    diff[p & ~g] = (0, 0, 255)
    diff[~p & g] = (255, 0, 0)
    panels = [overlay(pred, (0, 255, 0)), overlay(gt, (255, 0, 255)), diff]
    labels = ["Ours (classical)", "SAM2", "Diff: G=both R=ours B=SAM2"]
    for pnl, lab in zip(panels, labels):
        cv2.putText(pnl, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(pnl, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return np.hstack(panels)


# -----------------------------------------------------------------------------
# Main pipeline (importable for the web app)
# -----------------------------------------------------------------------------
def run(img, rect=None, method="grabcut", bg_points=None, bg_radius=45):
    """Return (mask, boundary_image) for a BGR image. Without a rect, the
    function uses a box with a 5% margin on each side."""
    H, W = img.shape[:2]
    if rect is None:
        mx, my = int(0.05 * W), int(0.05 * H)
        rect = (mx, my, W - 2 * mx, H - 2 * my)
    if method == "watershed":
        mask = segment_watershed(img, rect)
    else:
        mask = segment_grabcut(img, rect, bg_points=bg_points, radius=bg_radius)
    vis = img.copy()
    cv2.drawContours(vis, get_boundary(mask), -1, (0, 255, 0), 2)
    return mask, vis


def main():
    ap = argparse.ArgumentParser(description="Classical human boundary extraction (RGB)")
    ap.add_argument("--image", required=True, help="input RGB image")
    ap.add_argument("--rect", default=None, help="x,y,w,h box around the person (original image coords)")
    ap.add_argument("--method", default="grabcut", choices=["grabcut", "watershed"])
    ap.add_argument("--bg_points", default=None,
                    help="GrabCut only: clicks on non-person areas, as x1,y1;x2,y2 (original image coords)")
    ap.add_argument("--bg_radius", type=int, default=45,
                    help="radius in pixels of each background click disc")
    ap.add_argument("--sam_mask", default=None, help="SAM2 binary mask to compare against")
    ap.add_argument("--out_dir", default="results_q1")
    args = ap.parse_args()

    img_full = cv2.imread(args.image)
    if img_full is None:
        sys.exit(f"Could not read {args.image}")
    img, s = resize_max(img_full)

    # --- Get the bounding box: CLI, interactive, or default ---
    if args.rect:
        x, y, w, h = [int(round(int(v) * s)) for v in args.rect.split(",")]
        rect = (x, y, w, h)
    else:
        try:
            r = cv2.selectROI("Draw a box around the person, then ENTER", img, showCrosshair=False)
            cv2.destroyAllWindows()
            rect = tuple(int(v) for v in r) if r[2] > 0 and r[3] > 0 else None
        except cv2.error:
            rect = None  # no display available, so use the default box

    bg_points = None
    if args.bg_points:
        bg_points = [tuple(int(round(int(v) * s)) for v in p.split(","))
                     for p in args.bg_points.split(";")]

    mask, vis = run(img, rect, args.method, bg_points, max(1, int(round(args.bg_radius * s))))

    # --- Upscale results back to original resolution ---
    H0, W0 = img_full.shape[:2]
    mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
    vis = img_full.copy()
    cv2.drawContours(vis, get_boundary(mask), -1, (0, 255, 0), 2)

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_mask.png"), mask)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_boundary.png"), vis)
    print(f"Saved mask and boundary to {args.out_dir}/")

    # --- Optional SAM2 comparison ---
    if args.sam_mask:
        gt = cv2.imread(args.sam_mask, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            sys.exit(f"Could not read {args.sam_mask}")
        gt = cv2.resize(gt, (W0, H0), interpolation=cv2.INTER_NEAREST)
        gt = np.where(gt > 127, 255, 0).astype(np.uint8)
        metrics = compare_masks(mask, gt)
        print("\nComparison with SAM2:")
        for k, v in metrics.items():
            print(f"  {k:<16} {v:.4f}")
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_compare.png"),
                    make_comparison_image(img_full, mask, gt))


if __name__ == "__main__":
    main()
