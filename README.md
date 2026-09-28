# CSc 8830 Computer Vision: Assignments

**Author:** Liberty Ikpeogu

**Live web app:** _add your Streamlit link here after deploying_

Each assignment has its own folder with its own README. One Streamlit web app
at the top level shows every assignment, and its pages import the code from
the module folders, so the app and the command line give identical results.

| Module | Folder | Topic |
|---|---|---|
| 3 | [`module3/`](module3/) | Box and Gaussian blur by direct convolution and by FFT |
| 4 | [`module4/`](module4/) | Human boundaries in RGB and thermal images; Fourier-domain edges and segmentation |

## Run the web app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit opens the app at http://localhost:8501. Pick a module from the sidebar.

## Repository layout

```
streamlit_app.py     web app entry point and sidebar navigation
app_pages/           one page per assignment question
webapp_utils.py      helpers shared by the pages
module3/             Module 3 code, samples, results, report (and its original Flask app)
module4/             Module 4: q1/, q2/, q3/
requirements.txt     everything the web app needs
```

To add a module: create `moduleN/`, add its pages to `app_pages/`, and list
them in `streamlit_app.py`.
