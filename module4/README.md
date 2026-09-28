# CSc 8830 Computer Vision - Module 4

Boundaries and the Fourier domain.

**Author:** Liberty Ikpeogu

| Question | Code | What it does |
|---|---|---|
| Q1 | `q1/q1_rgb_human_boundary.py` | Finds a person's exact boundary in an RGB photo with GrabCut or marker-based watershed, then compares against SAM2. |
| Q2 | `q2/q2_thermal_human_boundary.py` | Finds a person's boundary in a thermal image with a white top-hat and hysteresis thresholding, then compares against SAM2. Decodes false-colour (rainbow) thermal images back to heat values. |
| Q3 | `q3/q3_fourier_edges_segmentation.pdf` | Derivations of edge detection and region segmentation in the Fourier domain. `fourier_lab.py` implements them and `q3_make_figures.py` rebuilds the figures. |

No deep learning or machine learning code runs in Q1 or Q2. SAM2 serves only as
the reference: I made its masks in Meta's SAM2 web demo, and they are stored in
each question's `sam2_masks/` folder.

## Q1 from the command line

```bash
cd q1
# Astronaut: box only
python q1_rgb_human_boundary.py --image images/astronaut.png --rect 15,5,360,507 \
    --sam_mask sam2_masks/astronaut_sam2.png
# Messi: box plus three background clicks
python q1_rgb_human_boundary.py --image images/messi5.jpg --rect 65,55,395,287 \
    --bg_points "110,300;170,320;360,310" --bg_radius 25 --sam_mask sam2_masks/messi5_sam2.png
# Zidane: box plus eight background clicks
python q1_rgb_human_boundary.py --image images/zidane.jpg --rect 115,195,1045,525 \
    --bg_points "860,300;850,520;880,680;950,620;1000,400;1050,500;1060,650;820,420" \
    --sam_mask sam2_masks/zidane_sam2.png
```

Add `--method watershed` to switch methods. Leave out `--rect` on a computer
with a display and a window lets you draw the box with the mouse.

## Q2 from the command line

```bash
cd q2
python q2_thermal_human_boundary.py --image images/llvip_010025_jet.png --rect 465,365,115,260 \
    --sam_mask sam2_masks/llvip_010025_sam2.png
python q2_thermal_human_boundary.py --image images/llvip_200007_jet.png --rect 1030,495,175,330 \
    --sam_mask sam2_masks/llvip_200007_sam2.png
python q2_thermal_human_boundary.py --image images/llvip_190003_jet.png --rect 350,685,110,290 \
    --sam_mask sam2_masks/llvip_190003_sam2.png
```

Add `--method otsu` for the plain Otsu baseline, or `--black_hot` for cameras
that show warm objects as dark.

## Results against SAM2

The full tables are in `q1/results/q1_metrics_vs_sam2.csv` and
`q2/results/q2_metrics_vs_sam2.csv`.

| Image | Method | IoU |
|---|---|---|
| Astronaut (RGB) | GrabCut | 0.81 |
| Messi (RGB) | GrabCut + 3 background clicks | 0.84 |
| Zidane (RGB) | GrabCut + 8 background clicks | 0.75 |
| LLVIP 010025 (thermal) | Top-hat + hysteresis | 0.96 |
| LLVIP 200007 (thermal) | Plain Otsu + hysteresis | 0.74 |
| LLVIP 190003 (thermal) | Top-hat + hysteresis | 0.84 |

## Image sources

- Astronaut: NASA portrait of Eileen Collins (public domain), via scikit-image.
- Messi and Zidane: sample images from the OpenCV and Ultralytics repositories.
- Thermal: LLVIP dataset (github.com/bupt-ai-cz/LLVIP). The `_jet.png` versions
  are the same heat data painted with the jet colour map.

## Web app

The Module 4 pages live in the course web app. From the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

