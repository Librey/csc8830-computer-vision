# Module 6: Optical Flow, Tracking and Structure from Motion

**Author:** Liberty Ikpeogu

Part A computes optical flow on two 30-second videos, measures what the flow reveals, and validates the Lucas–Kanade tracking equations on real frames. Part B reconstructs a flat book cover and four camera positions from four photos.

## Setup

```bash
cd module6
pip install -r requirements.txt
```

The two input videos (`Traffic_static_shot.mp4`, `Driving_pov.mp4`) exceed GitHub's file limits, so `.gitignore` keeps them out of the repo. Place them in `part_a/` before running Part A.

## Part A: optical flow and tracking

Run from `part_a/`, in this order:

```bash
# 1. dense optical flow videos + per-frame statistics  ->  output/
python optical_flow.py --video Traffic_static_shot.mp4 --start 5  --duration 30 --scale 0.5
python optical_flow.py --video Driving_pov.mp4         --start 10 --duration 30 --scale 0.5

# 2. evidence figures for "what optical flow tells us"  ->  figures/
python flow_analysis.py --traffic Traffic_static_shot.mp4 --driving Driving_pov.mp4 \
    --stats output --out figures

# 3. Lucas-Kanade from scratch vs measured pixel locations vs OpenCV  ->  validation/
python lk_validate.py --video Traffic_static_shot.mp4 --t 20 --moving_only
python lk_validate.py --video Driving_pov.mp4 --t 25 --min_rows 0.42 --max_rows 0.72
```

| File | What it does |
|---|---|
| `optical_flow.py` | Farnebäck dense flow; writes a side-by-side video (original, HSV flow, arrows) and a stats CSV |
| `flow_analysis.py` | Figures 1–6: segmentation, lane directions, focus of expansion, parallax, activity over time, texture failure |
| `lk_validate.py` | Pyramidal Lucas–Kanade coded in NumPy with bilinear interpolation; compares against patch-matched locations and `cv2.calcOpticalFlowPyrLK`; writes a 5×5 hand-worked example |
| `part_a_inference.md` | Write-up of what the flow shows, with figures |

Result: tracked points land within 0.42 px of their measured locations (mean 0.16 px traffic, 0.21 px driving) and within 0.05 px of OpenCV.

## Part B: structure from motion

```bash
cd part_b
python sfm_book.py --images images --out results
```

`sfm_book.py` computes K from the 24 mm-equivalent focal length, finds the cover corners, matches SIFT features, decomposes the homographies into camera motion, triangulates, runs bundle adjustment, and reports the recovered dimensions and camera positions in `results/results.txt`.

| Camera parameter | Value |
|---|---|
| Camera | Apple iPhone 17 Pro, 24 mm-equivalent |
| Image size | 1932 × 2576 px (downscaled from 3024 × 4032) |
| Focal length | 1786.1 px |
| Principal point | (966, 1288) px |

Result, with only the 14.5 cm width used for scale: length 20.05 cm (true 20.0 cm), corner angles 89.5° to 90.4°, mean reprojection error 0.68 px.

## Report

The full report with derivations, worked calculations and references is submitted as a PDF in Google Classroom.
