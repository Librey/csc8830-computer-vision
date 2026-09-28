# CSc 8830 Computer Vision - Module 3

Image blurring with a box or Gaussian filter, done two ways:

1. **Spatial domain:** direct convolution, written from the definition.
2. **Fourier domain:** multiply the FFT of the image by the FFT of the kernel, then take the inverse FFT.

The web app shows both results side by side. They match to within floating-point rounding (about 1e-12), which is the convolution theorem: convolution in space equals multiplication in frequency.

Repo: https://github.com/Librey/cv-module3-blur

## Setup

Requires Python 3.9 or newer.

```
pip install -r requirements.txt
```

## Run the web app

```
python app.py
```

Then open http://127.0.0.1:5000 in a browser. Choose a kernel (box or Gaussian), its size and sigma, and an image (sample or upload), then click "Blur and compare". The page shows the spatial blur, the Fourier blur, their difference, the spectra and error numbers. Unticking "Zero-pad before the FFT" shows the wrap-around error you get from circular convolution.

## Run from the command line

```
python blur_filters.py --input sample/cameraman.png --kernel gaussian --size 15 --sigma 2.5 --out results/
```

This prints the error between the two methods and saves the blurred images and spectra to `results/`.

## Notes

- Kernels are normalised so their weights sum to 1.
- The FFT method zero-pads the image and kernel to at least (H+K-1) x (W+K-1). Without padding, the DFT does circular convolution and the blur wraps around the edges.
- SciPy's `convolve2d` is only used as an independent check.
- Sample images come from scikit-image: `cameraman.png` is CC0 and `astronaut.png` is a NASA photo in the public domain.

Lecture reference: S. K. Nayar, First Principles of Computer Vision (Image Processing I and II), Columbia University.
