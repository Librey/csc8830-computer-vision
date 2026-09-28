#!/usr/bin/env python3
"""
make_report.py
==============
CSc 8830 Module 3 -- reproduces every experiment and figure in the report and
builds report/Module3_Report.pdf.

README
------
    pip install -r requirements.txt reportlab
    python make_report.py
Optional: --screens a.png b.png  embeds web-app screenshots;  --name "Your Name" --repo <github url>  fill the title block.

Output:  results/*.png (figures)   report/Module3_Report.pdf
"""

import argparse
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from blur_filters import (benchmark, fft_shape, box_kernel, conv2d_fft_circular_same, conv2d_fft_full,
                          conv2d_separable_full, conv2d_spatial_full, crop_same,
                          gaussian_kernel, gaussian_kernel_1d, load_image, to_uint8, validate)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
REP = os.path.join(HERE, "report")
os.makedirs(RES, exist_ok=True)
os.makedirs(REP, exist_ok=True)

# ---------------------------------------------------------------- fonts
ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(ttf, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(ttf, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(ttf, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(ttf, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(ttf, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.6, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=14.5, leading=18, spaceBefore=10, spaceAfter=6, alignment=0)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4, alignment=0)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.3, leading=11, textColor=colors.HexColor("#444444"), alignment=0, spaceAfter=10)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=20, leading=25, alignment=0, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10.5, textColor=colors.HexColor("#444444"), alignment=0)
CODE = ParagraphStyle("code", parent=BODY, fontName="DVM", fontSize=8, leading=11, alignment=0, backColor=colors.HexColor("#f2f3f5"), borderPadding=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.6, leading=11, alignment=0, spaceAfter=0)


def P(t, s=BODY):
    return Paragraph(t, s)


def eq(latex, size=13, scale=1.0):
    """Render a LaTeX-style equation to a PNG with matplotlib mathtext and return a reportlab image."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    buf.seek(0)
    w, h = Image.open(buf).size
    buf.seek(0)
    img = RLImage(buf, width=w / 220 * 72 * scale, height=h / 220 * 72 * scale)
    t = Table([[img]], hAlign="LEFT")
    t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 14)]))
    return t


def fig_img(path, width=6.5 * inch):
    w, h = Image.open(path).size
    return RLImage(path, width=width, height=width * h / w, hAlign="CENTER")


# ---------------------------------------------------------------- experiments
def experiments():
    out = {}
    cam = load_image(os.path.join(HERE, "sample", "cameraman.png"))
    ast = load_image(os.path.join(HERE, "sample", "astronaut.png"))
    out["cam_shape"], out["ast_shape"] = cam.shape, ast.shape

    # --- Fig 1: Gaussian and box, spatial vs fourier vs difference
    for name, ker in (("gaussian", gaussian_kernel(15, 2.5)), ("box", box_kernel(15))):
        v = validate(cam, ker)
        out[f"{name}15"] = v["metrics"]
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(cam, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original f")
        ax[1].imshow(v["spatial"], cmap="gray", vmin=0, vmax=255); ax[1].set_title("Spatial: f * h")
        ax[2].imshow(v["fourier"], cmap="gray", vmin=0, vmax=255); ax[2].set_title("Fourier: IFFT(F·H)")
        d = np.abs(v["diff"])
        im = ax[3].imshow(d, cmap="viridis", vmin=0, vmax=1e-9); ax[3].set_title("|difference|  (scale 0 to 1e-9)")
        for a in ax:
            a.axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        label = "Gaussian 15x15, sigma=2.5" if name == "gaussian" else "Box 15x15"
        fig.suptitle(f"{label}: max |spatial - Fourier| = {v['metrics']['max_abs_error']:.2e} gray levels", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, f"fig_{name}_compare.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # --- Fig 2: kernel spectra vs analytic transforms
    K, sig = 15, 2.5
    P_ = 512
    u = np.fft.fftshift(np.fft.fftfreq(P_))
    Hb = np.abs(np.fft.fftshift(np.fft.fft(np.full(K, 1.0 / K), P_)))            # 1-D box of width K, weights sum to 1
    g1 = gaussian_kernel_1d(K, sig)
    Hg = np.abs(np.fft.fftshift(np.fft.fft(g1, P_)))
    Hb_an = np.abs(np.sin(np.pi * u * K) / (K * np.sin(np.pi * u + 1e-15)))
    Hb_an[np.abs(u) < 1e-12] = 1.0
    Hg_an = np.exp(-2 * np.pi ** 2 * sig ** 2 * u ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(u, Hb, lw=2, label="numerical  |FFT of box|"); ax[0].plot(u, Hb_an, "--", lw=1.2, label=r"analytic  $|\sin(\pi u K)/(K\sin\pi u)|$")
    ax[0].set_title("Box kernel spectrum is a sinc (side lobes)"); ax[0].set_xlabel("u (cycles/pixel)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(u, Hg, lw=2, label="numerical  |FFT of Gaussian|"); ax[1].plot(u, Hg_an, "--", lw=1.2, label=r"analytic  $e^{-2\pi^2\sigma^2u^2}$")
    ax[1].set_title("Gaussian kernel spectrum is a Gaussian (no side lobes)"); ax[1].set_xlabel("u (cycles/pixel)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_kernel_spectra.png"), dpi=130); plt.close(fig)
    out["gauss_spec_err"] = float(np.max(np.abs(Hg - Hg_an)))

    # --- Fig 3: image spectrum * kernel spectrum
    H, W = cam.shape
    ker = gaussian_kernel(15, 2.5)
    shape = fft_shape(H, W, 15, 15)
    F = np.fft.fft2(cam, s=shape); Hk = np.fft.fft2(ker, s=shape)

    def sh(x):
        return np.log1p(np.abs(np.fft.fftshift(x)))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, s, t in zip(ax, (sh(F), np.abs(np.fft.fftshift(Hk)), sh(F * Hk)),
                       ("log|F(u,v)|  original image", "|H(u,v)|  Gaussian kernel", "log|F(u,v)·H(u,v)|  = log|G(u,v)|")):
        a.imshow(s, cmap="magma"); a.set_title(t, fontsize=10); a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_spectra_product.png"), dpi=130); plt.close(fig)

    # --- Fig 4: wrap-around without padding
    v_pad = validate(cam, ker, zero_pad=True)
    v_no = validate(cam, ker, zero_pad=False)
    d = np.abs(v_no["diff"])
    out["nopad_max"] = float(d.max())
    out["nopad_interior_max"] = float(d[8:-8, 8:-8].max())
    out["nopad_border_mean"] = float(np.mean(np.r_[d[:7].ravel(), d[-7:].ravel(), d[:, :7].ravel(), d[:, -7:].ravel()]))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].imshow(v_no["fourier"], cmap="gray", vmin=0, vmax=255); ax[0].set_title("FFT blur, no zero-padding"); ax[0].axis("off")
    im = ax[1].imshow(d, cmap="inferno"); ax[1].set_title("|spatial - FFT no pad|  (gray levels)"); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].semilogy(np.maximum(d.max(axis=0), 1e-16)); ax[2].set_title("worst error per column"); ax[2].set_xlabel("column"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_wraparound.png"), dpi=130); plt.close(fig)

    # --- Table: errors across kernels / sizes / images
    rows = []
    for imgname, img in (("cameraman (gray)", cam), ("astronaut (RGB)", ast)):
        for kind, K_, s_ in (("box", 3, None), ("box", 9, None), ("box", 21, None),
                             ("gaussian", 5, 1.0), ("gaussian", 15, 2.5), ("gaussian", 31, 5.0)):
            kk = box_kernel(K_) if kind == "box" else gaussian_kernel(K_, s_)
            m = validate(img, kk)["metrics"]
            rows.append((imgname, f"{kind} {K_}x{K_}" + (f", s={s_}" if s_ else ""), m["max_abs_error"], m["rmse"], m["spatial_vs_scipy_max_abs"]))
    out["table"] = rows

    # --- Separable vs direct
    g1 = gaussian_kernel_1d(15, 2.5)
    out["sep_err"] = float(np.max(np.abs(conv2d_spatial_full(cam, np.outer(g1, g1)) - conv2d_separable_full(cam, g1))))

    # --- Timing
    sizes = [3, 5, 9, 15, 21, 31, 41]
    bench = benchmark(cam, sizes, sigma_fn=lambda k: k / 6, repeats=2)
    out["bench"] = bench
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for key, lab in (("direct", "spatial, direct (K² multiplies/pixel)"), ("separable", "spatial, separable (2K per pixel)"), ("fft", "Fourier (FFT, size independent of K)")):
        ax.plot(sizes, [r[key] * 1e3 for r in bench], "o-", label=lab)
    ax.set_yscale("log"); ax.set_xlabel("kernel size K"); ax.set_ylabel("time for 512x512 image (ms)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); ax.set_title("Cost versus kernel size (this implementation, NumPy)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_timing.png"), dpi=130); plt.close(fig)

    # --- 1-D worked example
    f = np.array([2, 4, 1, 3.0]); h = np.array([1, 2, 1.0])
    full = np.convolve(f, h)
    N = len(f) + len(h) - 1
    Fk = np.fft.fft(f, N); Hk1 = np.fft.fft(h, N); Gk = Fk * Hk1
    back = np.real(np.fft.ifft(Gk))
    N4 = len(f)
    circ = np.real(np.fft.ifft(np.fft.fft(f, N4) * np.fft.fft(h, N4)))
    out["ex"] = dict(f=f, h=h, full=full, N=N, F=Fk, H=Hk1, G=Gk, back=back, circ=circ)
    return out


# ---------------------------------------------------------------- PDF
def fmt_c(z):
    re, im = np.round(z.real, 3) + 0.0, np.round(z.imag, 3) + 0.0
    if abs(im) < 5e-4:
        return f"{re:.3f}"
    return f"{re:.3f} {'+' if im >= 0 else '-'} {abs(im):.3f}j"


def build_pdf(o, screens, name="(your name)", repo="https://github.com/(your-username)/cv-module3-blur"):
    path = os.path.join(REP, "Module3_Report.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="CSc 8830 Module 3: Blurring in the spatial and Fourier domains",
                            author=name)
    s = []
    s += [P("Image Blurring by Spatial Filtering and its Fourier-Domain Equivalent", TITLE),
          P("CSc 8830 Computer Vision &nbsp;|&nbsp; Module 3 Assignment", SUB),
          P(f"Student: <b>{name}</b> &nbsp;&nbsp; GitHub repository: <b>{repo}</b>", SUB),
          Spacer(1, 10)]

    s += [P("1. Summary", H1),
          P(f"I blurred images with a normalised box filter and a Gaussian filter in two ways: (a) direct convolution in the spatial "
            f"domain, coded from the definition without any library convolution, and (b) multiplication of Fourier transforms followed by an "
            f"inverse transform. Across every kernel tested, the two results agree to within floating-point rounding: the largest difference "
            f"for a 15×15 Gaussian on a 512×512 image is <b>{o['gaussian15']['max_abs_error']:.2e}</b> gray levels on a 0 to 255 scale, "
            f"about twelve orders of magnitude below one quantisation step (one gray level). Section 3 proves the convolution theorem and Sections 4 to 6 "
            f"validate it experimentally. The one condition that must hold is zero-padding before the FFT; without it the two "
            f"methods disagree near the borders by up to <b>{o['nopad_max']:.0f}</b> gray levels.")]

    # ---- implementation
    s += [P("2. Implementation", H1),
          P("Everything is in <font name='DVM'>blur_filters.py</font> (with a ReadMe header) and is exposed through the Flask web app "
            "<font name='DVM'>app.py</font>. Both images below use the same kernels and the same zero-padding assumption (pixels outside the image are 0)."),
          P("<b>Kernels.</b> The box filter has all K² weights equal to 1/K². The Gaussian is "
            "h[m,n] ∝ exp(−(m²+n²)/2σ²), sampled on a K×K grid and normalised. Both kernels sum to 1, so mean brightness is preserved "
            "(an un-normalised 5×5 box would make the image 25× brighter and saturate it)."),
          P("<b>Spatial filtering.</b> Convolution is implemented straight from its definition:"),
          eq(r"g[i,j]\;=\;\sum_{a}\sum_{b} h[a,b]\,f[i-a,\;j-b]"),
          P("Each kernel weight h[a,b] contributes a copy of the image shifted by (a,b); adding the K² shifted copies is exactly the "
            "flip-and-slide sum (the flip is built into the index i−a, j−b). The full result has size (H+K−1)×(W+K−1) and is cropped "
            f"back to H×W. As a check, the Gaussian is separable, and two 1-D passes give the same answer as the 2-D kernel to "
            f"{o['sep_err']:.1e}."),
          P("<b>Fourier filtering.</b> Zero-pad the image and the kernel to the same size (at least (H+K−1)×(W+K−1); the code rounds up to an FFT-friendly size), take the 2-D FFT of each, multiply "
            "them element by element, and take the inverse FFT. In code: "),
          P("F = fft2(f, s=shape); Hk = fft2(h, s=shape); g = real(ifft2(F * Hk))", CODE),
          Spacer(1, 4),
          P("<b>Independent reference.</b> SciPy's <font name='DVM'>convolve2d</font> is used only as a third opinion; it agrees with both methods "
            "(Table 1).")]

    # ---- theory
    s += [P("3. Theory: convolution in space equals multiplication in frequency", H1),
          P("<b>3.1 Continuous form.</b> Let g(x) = (f ∗ h)(x) = ∫ f(τ) h(x−τ) dτ and let the Fourier transform be "
            "F(u) = ∫ f(x) e<super>−j2πux</super> dx. Then"),
          eq(r"G(u)=\int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} f(\tau)\,h(x-\tau)\,e^{-j2\pi u x}\,d\tau\,dx"),
          P("Exchange the order of integration and write e<super>−j2πux</super> = e<super>−j2πuτ</super> · e<super>−j2πu(x−τ)</super>:"),
          eq(r"G(u)=\int f(\tau)\,e^{-j2\pi u\tau}\,d\tau\;\cdot\;\int h(x-\tau)\,e^{-j2\pi u (x-\tau)}\,dx"),
          P("In the second integral substitute μ = x−τ. Because τ is finite the limits stay (−∞, ∞), so it equals H(u), the transform of h. "
            "The first integral is F(u). Therefore"),
          eq(r"G(u)=F(u)\,H(u)\qquad\Longleftrightarrow\qquad g=f\ast h", size=14),
          P("The same derivation in two dimensions gives G(u,v) = F(u,v) H(u,v). Running the argument in the other direction shows that a product "
            "in space is a convolution in frequency."),
          P("<b>3.2 Discrete form (what the code uses).</b> For P×Q arrays, the DFT is "
            "F[u,v] = Σ<sub>m</sub>Σ<sub>n</sub> f[m,n] e<super>−j2π(um/P+vn/Q)</super>, and the product of DFTs corresponds to "
            "<i>circular</i> convolution:"),
          eq(r"g_c[i,j]=\sum_{m=0}^{P-1}\sum_{n=0}^{Q-1} f[m,n]\;h[(i-m)\;\mathrm{mod}\;P,\;(j-n)\;\mathrm{mod}\;Q]"),
          P("Take its DFT, swap the sums, and put i′ = (i−m) mod P and j′ = (j−n) mod Q. As i runs over 0…P−1, so does i′ (a bijection), and the "
            "complex exponential is periodic in P, so e<super>−j2πui/P</super> = e<super>−j2πui′/P</super> · e<super>−j2πum/P</super>. The sum splits:"),
          eq(r"G_c[u,v]=\left(\sum_{m,n} f[m,n]\,e^{-j2\pi(\frac{um}{P}+\frac{vn}{Q})}\right)\left(\sum_{i',j'} h[i',j']\,e^{-j2\pi(\frac{ui'}{P}+\frac{vj'}{Q})}\right)=F[u,v]\,H[u,v]"),
          P("<b>3.3 Why zero-padding makes it equal to spatial filtering.</b> An image of height H and a kernel of height K, both zero outside "
            "their support, are padded to P ≥ H+K−1 rows. In the circular sum, whenever i−m &lt; 0 the index wraps to (i−m)+P ≥ P−H+1 ≥ K, and "
            "h is zero at indices ≥ K. So every wrapped term is zero, the circular sum equals the ordinary sum, and "
            "IFFT(F·H) = f ∗ h exactly (same argument for columns). Padding to a smaller size lets the tail of the blur wrap around to the opposite "
            "edge, which is what Section 5.3 shows."),
          P("<b>3.4 What the two kernels look like in frequency.</b> The transform of a box of width K is a (discrete) sinc "
            "|sin(πuK) / (K sin πu)|, with side lobes. The transform of a Gaussian of spread σ pixels is another Gaussian, "
            "H(u) = exp(−2π²σ²u²), of spread 1/(2πσ) in frequency, so a wider blur in space is a narrower low-pass in frequency. Both are low-pass "
            "filters; multiplying by them attenuates high frequencies, which are the edges and fine detail, and this is what blurring is.")]

    # ---- by example
    ex = o["ex"]
    s += [PageBreak(), P("3.5 Worked example (by hand, checked by computer)", H2),
          P("Signal f = [2, 4, 1, 3] and kernel h = [1, 2, 1] (un-normalised so all numbers are integers). "
            "<b>Spatial:</b> g[i] = Σ<sub>a</sub> h[a] f[i−a]."),
          P("g[0] = 1·2 = 2<br/>g[1] = 1·4 + 2·2 = 8<br/>g[2] = 1·1 + 2·4 + 1·2 = 11<br/>g[3] = 1·3 + 2·1 + 1·4 = 9<br/>"
            "g[4] = 2·3 + 1·1 = 7<br/>g[5] = 1·3 = 3<br/>"
            "<b>g = [2, 8, 11, 9, 7, 3]</b> (length 4+3−1 = 6).", CODE),
          Spacer(1, 6),
          P(f"<b>Fourier:</b> pad both to N = {ex['N']}, take the 6-point DFT of each, multiply, and invert. "
            "The table lists each frequency k.")]
    data = [["k", "F[k]", "H[k]", "G[k] = F[k]·H[k]"]]
    for k in range(ex["N"]):
        data.append([str(k), fmt_c(ex["F"][k]), fmt_c(ex["H"][k]), fmt_c(ex["G"][k])])
    t = Table(data, colWidths=[0.5 * inch, 1.7 * inch, 1.7 * inch, 2.0 * inch])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "DV-B"), ("FONTNAME", (0, 1), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf4")),
                           ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    s += [t, Spacer(1, 6),
          P("Inverse DFT of G: " + "[" + ", ".join(f"{x:.3f}" for x in ex["back"]) + "] — identical to the spatial result [2, 8, 11, 9, 7, 3].", BODY),
          P("<b>Without padding</b> (N = 4, the length of f): the circular result is [" + ", ".join(f"{x:.0f}" for x in ex["circ"]) +
            "]. The last two spatial outputs (7 and 3) wrapped around and were added to positions 0 and 1 (2+7 = 9, 8+3 = 11). This is the same effect "
            "that appears at image borders in Section 5.3.")]

    # ---- experiments
    s += [P("4. Experiment: spatial versus Fourier result", H1),
          P(f"Test image: 512×512 grayscale “cameraman” (from scikit-image). The figure shows the original, the spatially blurred image, the Fourier-blurred image, "
            f"and the absolute difference on a scale of 0 to 10<super>−9</super> gray levels. The two blurred images are visually identical and the difference is "
            f"rounding noise."),
          KeepTogether([fig_img(os.path.join(RES, "fig_gaussian_compare.png")),
                        fig_img(os.path.join(RES, "fig_box_compare.png")),
                        P("Figure 1. Gaussian (top) and box (bottom) blur of the same image, computed both ways.", CAP)])]

    rows = [["Image", "Kernel", "max |spatial−FFT|", "RMSE", "max |spatial−SciPy|"]]
    for r in o["table"]:
        rows.append([r[0], r[1], f"{r[2]:.2e}", f"{r[3]:.2e}", f"{r[4]:.2e}"])
    t = Table(rows, repeatRows=1, colWidths=[1.45 * inch, 1.6 * inch, 1.3 * inch, 1.0 * inch, 1.35 * inch])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "DV-B"), ("FONTNAME", (0, 1), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), 8),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf4")),
                           ("ALIGN", (2, 1), (-1, -1), "RIGHT")]))
    s += [KeepTogether([t, P("Table 1. Errors in gray levels (0–255 scale) for several kernels and sizes, on a grayscale and an RGB image (each channel filtered separately). "
                             f"Every entry is at most {max(r[2] for r in o['table']):.1e}, i.e. floating-point rounding (the errors grow slightly with kernel size because more terms are summed).", CAP)])]

    s += [P("5. Evidence from the frequency domain", H1), P("<b>5.1 The kernels are low-pass filters.</b> "
                                                            f"The numerically computed spectra of the 15×15 kernels match the analytic transforms from Section 3.4 "
                                                            f"(largest Gaussian deviation {o['gauss_spec_err']:.1e}, from truncating the Gaussian to 15 taps). The box has side lobes; the Gaussian does not, "
                                                            f"which is why box blur looks blocky (it leaks some high frequencies back in) and Gaussian blur looks natural."),
          fig_img(os.path.join(RES, "fig_kernel_spectra.png")), P("Figure 2. Kernel spectra (numerical solid, analytic dashed).", CAP),
          P("<b>5.2 Multiplying the spectra is the blur.</b> Left: spectrum of the original image, concentrated at low frequencies with strong axis-aligned streaks from "
            "the image edges. Middle: the Gaussian's spectrum, a small bright disc around zero frequency. Right: their product, which keeps the low frequencies "
            "and suppresses the rest. The inverse transform of the right panel is the blurred image."),
          fig_img(os.path.join(RES, "fig_spectra_product.png")), P("Figure 3. F(u,v), H(u,v) and G(u,v) = F(u,v)H(u,v), log-magnitude, zero frequency at the centre.", CAP)]

    s += [P("<b>5.3 What goes wrong without zero-padding.</b> If the FFT is done at the image size (no padding), the result is a circular convolution. "
            f"Interior pixels still match to {o['nopad_interior_max']:.1e}, but pixels within half a kernel width of an edge are blended with the opposite side of the image: "
            f"the largest error is {o['nopad_max']:.1f} gray levels and the mean error in the 7-pixel border is {o['nopad_border_mean']:.2f}. "
            "This confirms the condition derived in Section 3.3."),
          fig_img(os.path.join(RES, "fig_wraparound.png")), P("Figure 4. Un-padded FFT blur: error is confined to the borders.", CAP)]

    s += [P("6. Cost: when is the Fourier route worth it?", H1),
          P("The direct spatial method needs K² multiplications per pixel (2K if separable); the FFT route costs about the same whatever K is, because the "
            "transform size only depends on the image. Timings for this NumPy/SciPy implementation on a 512×512 image are below; absolute numbers depend on the machine, and "
            "a compiled or vectorised convolution would move the crossover point, but the trend is the general one."),
          fig_img(os.path.join(RES, "fig_timing.png"), width=5.0 * inch), P("Figure 5. Time versus kernel size.", CAP)]
    b = o["bench"]
    s += [P(f"For K = 3 the direct method takes {b[0]['direct']*1e3:.0f} ms and the FFT {b[0]['fft']*1e3:.0f} ms; for K = 41 the direct method takes "
            f"{b[-1]['direct']*1e3:.0f} ms versus {b[-1]['fft']*1e3:.0f} ms. The separable Gaussian ({b[-1]['separable']*1e3:.0f} ms at K = 41) is far cheaper than the 2-D direct "
            f"version, as the lecture notes on separability predict, but it still grows linearly with K while the FFT stays flat.")]

    if screens:
        s += [PageBreak(), P("7. Web application demonstration", H1),
              P("The web app (<font name='DVM'>python app.py</font>, then http://127.0.0.1:5000) lets the user pick the kernel, size, sigma, and image, and shows the spatial result, "
                "the Fourier result, their difference, the spectra, and the error metrics. A checkbox disables zero-padding to reproduce the wrap-around error. "
                "A screen recording of the working system is submitted with this report.")]
        for i, sp in enumerate(screens, 1):
            s += [fig_img(sp, width=6.0 * inch), P(f"Figure {5 + i}. Screenshot of the web application.", CAP)]

    s += [P("8. Conclusions", H1),
          P("(1) Blurring with a normalised box or Gaussian kernel is a linear shift-invariant operation, i.e. a convolution. "
            "(2) The convolution theorem G = F·H, proved above in continuous and discrete form, means the same blur can be computed by multiplying spectra. "
            f"(3) Experimentally, the two routes agree to within {max(r[2] for r in o['table']):.1e} gray levels for every kernel tested, which is floating-point rounding. "
            "(4) Equality holds for the ordinary (linear) convolution only if both arrays are zero-padded to at least (H+K−1)×(W+K−1); otherwise the DFT's circular "
            "convolution wraps the blur across the image edges. (5) The Fourier view also explains <i>why</i> the blur looks the way it does: the kernels are low-pass "
            "filters, and the Gaussian's smooth spectrum avoids the side-lobe artifacts of the box.")]

    s += [P("References", H2),
          P("Nayar, S. K., <i>Image Processing I</i> (FPCV-1-4) and <i>Image Processing II</i> (FPCV-1-5), First Principles of Computer Vision, Columbia University, 2022. "
            "The kernels, the convolution definition and the convolution-theorem derivation follow these lectures. The zero-padding condition in 3.3 is an addition needed for the discrete case.", CAP)]

    doc.build(s, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(0.85 * inch, 0.5 * inch, "CSc 8830 Computer Vision - Module 3")
    canvas.drawRightString(letter[0] - 0.85 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", nargs="*", default=[])
    ap.add_argument("--name", default="(your name)")
    ap.add_argument("--repo", default="https://github.com/(your-username)/cv-module3-blur")
    a = ap.parse_args()
    o = experiments()
    print("pdf ->", build_pdf(o, a.screens, a.name, a.repo))
