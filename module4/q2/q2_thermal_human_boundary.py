"""
===============================================================================
CSc 8830 Computer Vision, Module 4, Question 2
Human boundary extraction from a thermal (long-wave infrared) image using
classical OpenCV methods (no deep learning or machine learning)
Author: Liberty Ikpeogu
===============================================================================

README
------
Install:
    pip install opencv-python numpy

Run:
    # Draw a box around the person (or people) with the mouse, then ENTER
    python q2_thermal_human_boundary.py --image llvip_200007.jpg

    # Pass the box as x,y,w,h instead
    python q2_thermal_human_boundary.py --image llvip_200007.jpg --rect 1030,495,175,330

    # Compare against a SAM2 mask (binary PNG, person in white)
    python q2_thermal_human_boundary.py --image llvip_200007.jpg --rect 1030,495,175,330 \\
        --sam_mask llvip_200007_sam2.png

    # The other two test images
    python q2_thermal_human_boundary.py --image llvip_010025.jpg --rect 465,365,115,260
    python q2_thermal_human_boundary.py --image llvip_190003.jpg --rect 350,685,110,290

    # Plain Otsu threshold instead of top-hat + Otsu
    python q2_thermal_human_boundary.py --image llvip_200007.jpg --method otsu

    # False-colour (rainbow / iron) images work too: the script matches each
    # pixel's colour to its palette position to recover the heat value.
    # It detects the palette itself; --palette jet forces one.
    python q2_thermal_human_boundary.py --image llvip_200007_jet.png --rect 1030,495,175,330

    # Camera in "black-hot" mode (warm objects dark): invert first
    python q2_thermal_human_boundary.py --image frame.png --black_hot

The script writes to --out_dir (default ./results_q2):
    <name>_mask.png        binary mask, 255 marks the person
    <name>_boundary.png    thermal image with the boundary drawn in white
    <name>_compare.png     our mask, the SAM2 mask, and a difference map
                           (written only when you pass --sam_mask)
It prints IoU, Dice, precision, recall and boundary F-score to the console.

Why thermal differs from RGB
----------------------------
A thermal camera measures emitted heat, not reflected light. Skin sits near
33 C, so people show up brighter than a cold night background regardless of
clothing colour or lighting. Intensity alone carries most of the signal, and
the task reduces to separating "warm and compact" from "background".
Two things break that: backgrounds that stored heat during the day (pavement,
walls, car engines) and clothing that insulates, which makes a coat colder
than exposed skin.

Methods
-------
tophat (default)
    1. Bilateral filter: smooths sensor noise but keeps the body edge sharp.
    2. White top-hat: T = I - open(I, B), where B is a disc wider than the
       person. The opening erases anything narrower than B, so open(I, B)
       estimates the slowly varying background heat. Subtracting it leaves
       compact warm objects on a flat, near-zero background.
    3. Hysteresis threshold on T inside the box. Otsu's threshold (which
       maximises between-class variance) marks sure-person pixels. Pixels
       above a lower threshold join them only if connected, which recovers
       insulated clothing that runs colder than skin.
    4. A closing sized to the body bridges thin cold bands (belts, collars).
       Then morphological cleanup, keep large components, fill holes, and
       cv2.findContours traces the exact boundary.

otsu
    Steps 1, 3 and 4 on the raw smoothed intensities, without the top-hat. This works
    on cold backgrounds and fails once the background is warm, which makes it
    a useful baseline for the report.
===============================================================================
"""

import argparse
import os
import sys

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Mask cleanup helpers (same logic as Question 1)
# -----------------------------------------------------------------------------
def keep_large_components(mask, keep_ratio=0.15):
    """Keep the largest blob plus any blob at least `keep_ratio` of its size,
    so two people in one box both survive while specks disappear."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]  # label 0 is the background
    keep = 1 + np.flatnonzero(areas >= keep_ratio * areas.max())
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


def fill_holes(mask):
    """Fill interior holes by flood-filling the background from a corner."""
    h, w = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def clean_mask(mask, k=3):
    """Open to remove specks, close to bridge small gaps (a cold belt or
    strap can split a body in two), keep large blobs, fill holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = keep_large_components(mask)
    return fill_holes(mask)


def get_boundary(mask):
    """Outer contours with every boundary pixel kept (CHAIN_APPROX_NONE)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------
PALETTES = {
    "jet": cv2.COLORMAP_JET,          # blue -> green -> red (classic rainbow)
    "rainbow": cv2.COLORMAP_RAINBOW,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,  # close to the "iron" look of many cameras
    "hot": cv2.COLORMAP_HOT,
}


def palette_lut(name):
    """The 256 BGR colours of a colour map, index 0 = coldest."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    return cv2.applyColorMap(ramp, PALETTES[name]).reshape(256, 3).astype(np.float32)


def is_false_colour(img):
    """True when the three channels differ, meaning a palette painted the image.
    Grayscale thermal images saved as 3 channels have equal channels."""
    if img.ndim != 3:
        return False
    diff = np.abs(img[..., 0].astype(int) - img[..., 2].astype(int))
    return np.percentile(diff, 90) > 20


def guess_palette(img, n=4000):
    """Pick the palette whose colours sit closest to a sample of the image."""
    rng = np.random.default_rng(0)
    px = img.reshape(-1, 3)[rng.integers(0, img.shape[0] * img.shape[1], n)].astype(np.float32)
    best, best_err = None, np.inf
    for name in PALETTES:
        lut = palette_lut(name)
        d = ((px[:, None, :] - lut[None, :, :]) ** 2).sum(2).min(1)
        if d.mean() < best_err:
            best, best_err = name, d.mean()
    return best


def decode_palette(img, palette="auto"):
    """Turn a false-colour thermal image back into heat values 0-255.
    Each pixel gets the index of its nearest palette colour. A plain
    grayscale conversion fails here: in the rainbow palette, green comes out
    brighter than red even though red is hotter."""
    if palette == "auto":
        palette = guess_palette(img)
    lut = palette_lut(palette)
    colours, inverse = np.unique(img.reshape(-1, 3), axis=0, return_inverse=True)
    colours = colours.astype(np.float32)
    idx = np.empty(len(colours), np.uint8)
    for s in range(0, len(colours), 20000):  # chunks keep memory small
        c = colours[s:s + 20000]
        idx[s:s + 20000] = ((c[:, None, :] - lut[None, :, :]) ** 2).sum(2).argmin(1)
    return idx[inverse.ravel()].reshape(img.shape[:2]), palette


def to_gray(img, black_hot=False, palette="auto"):
    """Return one channel of heat values where warm = bright.
    Handles three kinds of input: grayscale, grayscale stored as 3 identical
    channels, and false-colour (palette) images, which get decoded."""
    if img.ndim == 3 and is_false_colour(img):
        gray, _ = decode_palette(img, palette)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if gray.dtype != np.uint8:  # 16-bit radiometric data: stretch to 8 bits
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return 255 - gray if black_hot else gray


def segment_thermal(gray, rect, method="tophat", low_frac=0.5):
    """Return a binary mask (255 = person) for the people inside `rect`."""
    x, y, w, h = rect
    H, W = gray.shape

    # 1. Edge-preserving smoothing
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=25, sigmaSpace=7)

    if method == "tophat":
        # 2. White top-hat. The disc must be wider than a person, so we size it
        #    from the box's shorter side (a box around a standing person is
        #    about one person wide). We compute it on a padded crop so the
        #    opening sees real background around the box edges.
        k = int(1.2 * min(w, h)) | 1  # force an odd size
        k = max(k, 15)
        pad = k
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = smooth[y0:y1, x0:x1]
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, disc)
        signal = np.zeros_like(gray)
        signal[y0:y1, x0:x1] = tophat
    else:
        signal = smooth

    # 3. Hysteresis threshold inside the box, the same idea Canny uses for
    #    edges. Otsu's threshold t_high marks pixels that are surely the
    #    person. Insulated clothing (a coat hem, a hood) runs colder, so we
    #    also accept pixels above a lower threshold t_low, but only when they
    #    connect to a sure pixel. Background warm spots stay out because they
    #    do not touch the body.
    roi = signal[y:y + h, x:x + w]
    t_high, strong = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_level = np.percentile(roi, 20)            # typical background value in the box
    t_low = bg_level + low_frac * (t_high - bg_level)
    weak = (roi >= t_low).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    touching = np.unique(labels[strong > 0])
    touching = touching[touching > 0]
    roi_mask = np.where(np.isin(labels, touching), 255, 0).astype(np.uint8)

    mask = np.zeros((H, W), np.uint8)
    mask[y:y + h, x:x + w] = roi_mask

    # 4. Closing scaled to the body (about 1/25 of the box height) bridges thin
    #    cold bands such as a belt or collar, then the usual cleanup.
    kb = max(3, int(h / 25)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
    return clean_mask(mask)


# -----------------------------------------------------------------------------
# Evaluation against SAM2 (same metrics as Question 1)
# -----------------------------------------------------------------------------
def boundary_f_score(pred, gt, tol=3):
    """Boundary F-score. Precision counts our boundary pixels within `tol` px
    of the SAM2 boundary; recall counts the reverse."""
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
    """Compute region and boundary metrics between our mask and SAM2's."""
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


def false_colour(gray):
    """Thermal images read better in a heat colour map (dark blue = cold,
    yellow = hot)."""
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def make_comparison_image(gray, pred, gt, crop=None, base=None):
    """Three panels: our boundary, SAM2's boundary, and a difference map
    (green for both, red for ours only, blue for SAM2 only). `crop` zooms
    into a region so small, distant people stay visible."""
    if base is None:
        base = false_colour(gray)

    def overlay(mask, color):
        out = base.copy()
        cv2.drawContours(out, get_boundary(mask), -1, color, 2)
        return out

    diff = np.zeros_like(base)
    p, g = pred > 127, gt > 127
    diff[p & g] = (0, 200, 0)
    diff[p & ~g] = (0, 0, 255)
    diff[~p & g] = (255, 0, 0)
    panels = [overlay(pred, (0, 255, 0)), overlay(gt, (255, 0, 255)), diff]
    if crop is not None:
        cx, cy, cw, ch = crop
        panels = [p_[cy:cy + ch, cx:cx + cw] for p_ in panels]
        scale = 400 / ch
        panels = [cv2.resize(p_, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                  for p_ in panels]
    labels = ["Ours (classical)", "SAM2", "G=both R=ours B=SAM2"]
    for pnl, lab in zip(panels, labels):
        cv2.putText(pnl, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(pnl, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.hstack(panels)


# -----------------------------------------------------------------------------
# Main pipeline (importable for the web app)
# -----------------------------------------------------------------------------
def run(img, rect=None, method="tophat", black_hot=False, palette="auto"):
    """Return (mask, boundary_image) for a thermal image.
    Without a rect, the function uses the whole image. False-colour inputs
    keep their own colours in the output; grayscale inputs get INFERNO."""
    gray = to_gray(img, black_hot, palette)
    H, W = gray.shape
    if rect is None:
        rect = (0, 0, W, H)
    mask = segment_thermal(gray, rect, method)
    vis = img.copy() if (img.ndim == 3 and is_false_colour(img)) else false_colour(gray)
    cv2.drawContours(vis, get_boundary(mask), -1, (255, 255, 255), 3)
    cv2.drawContours(vis, get_boundary(mask), -1, (0, 0, 0), 1)
    return mask, vis


def main():
    ap = argparse.ArgumentParser(description="Classical human boundary extraction (thermal)")
    ap.add_argument("--image", required=True, help="input thermal image")
    ap.add_argument("--rect", default=None, help="x,y,w,h box around the person or people")
    ap.add_argument("--method", default="tophat", choices=["tophat", "otsu"])
    ap.add_argument("--black_hot", action="store_true", help="invert a black-hot image")
    ap.add_argument("--palette", default="auto", choices=["auto"] + list(PALETTES),
                    help="colour map of a false-colour image (auto-detected by default)")
    ap.add_argument("--sam_mask", default=None, help="SAM2 binary mask to compare against")
    ap.add_argument("--out_dir", default="results_q2")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read {args.image}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Get the box: CLI, interactive, or whole image
    rect = None
    if args.rect:
        rect = tuple(int(v) for v in args.rect.split(","))
    else:
        try:
            shown = img if img.dtype == np.uint8 else to_gray(img)
            r = cv2.selectROI("Draw a box around the person, then ENTER", shown, showCrosshair=False)
            cv2.destroyAllWindows()
            rect = tuple(int(v) for v in r) if r[2] > 0 and r[3] > 0 else None
        except cv2.error:
            rect = None  # no display available, so use the whole image

    mask, vis = run(img, rect, args.method, args.black_hot, args.palette)

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_mask.png"), mask)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_boundary.png"), vis)
    print(f"Saved mask and boundary to {args.out_dir}/")

    if args.sam_mask:
        gt = cv2.imread(args.sam_mask, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            sys.exit(f"Could not read {args.sam_mask}")
        gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = np.where(gt > 127, 255, 0).astype(np.uint8)
        metrics = compare_masks(mask, gt)
        print("\nComparison with SAM2:")
        for k, v in metrics.items():
            print(f"  {k:<16} {v:.4f}")
        # Zoom the comparison onto the box (plus a margin) so people stay visible
        crop = None
        if rect is not None:
            x, y, w, h = rect
            m = int(0.15 * max(w, h))
            H, W = mask.shape
            x0, y0 = max(0, x - m), max(0, y - m)
            crop = (x0, y0, min(W, x + w + m) - x0, min(H, y + h + m) - y0)
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_compare.png"),
                    make_comparison_image(to_gray(img, args.black_hot, args.palette), mask, gt, crop,
                                          base=img.copy() if is_false_colour(img) else None))


if __name__ == "__main__":
    main()
