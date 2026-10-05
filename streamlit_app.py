"""
CSc 8830 Computer Vision: assignment demos (Liberty Ikpeogu)

README
------
Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
Then open the address Streamlit prints (usually http://localhost:8501).

Each module gets its own section in the sidebar. To add a module, create
its page files in app_pages/ and list them below.
"""
import streamlit as st

st.set_page_config(page_title="CSc 8830 Computer Vision", page_icon="👁️", layout="wide")

pages = {
    "Course": [
        st.Page("app_pages/home.py", title="Home", icon="🏠", default=True),
    ],
    "Module 3": [
        st.Page("app_pages/m3_blur.py", title="Blur: spatial vs Fourier", icon="🌫️"),
    ],
    "Module 4": [
        st.Page("app_pages/m4_q1_rgb.py", title="Q1: Human boundary, RGB", icon="🧍"),
        st.Page("app_pages/m4_q2_thermal.py", title="Q2: Human boundary, thermal", icon="🌡️"),
        st.Page("app_pages/m4_q3_fourier.py", title="Q3: Fourier-domain edges", icon="〰️"),
    ],
    "Module 6": [
        st.Page("app_pages/m6_flow.py", title="Optical flow", icon="🌊"),
        st.Page("app_pages/m6_tracking.py", title="Lucas–Kanade tracking", icon="🎯"),
        st.Page("app_pages/m6_sfm.py", title="Structure from motion", icon="📐"),
    ],
}
st.navigation(pages).run()
