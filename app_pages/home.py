import streamlit as st

st.title("CSc 8830 Computer Vision")
st.write("Assignment demos by **Liberty Ikpeogu**. Every page runs the same Python "
         "code that lives in the GitHub repository, so what you see here is what "
         "the scripts produce.")

st.subheader("Module 3: blur in space and in frequency")
st.page_link("app_pages/m3_blur.py", label="Box and Gaussian blur, computed by direct convolution and by FFT",
             icon="🌫️")
st.caption("Shows that both methods agree to rounding error, and what goes wrong without zero-padding.")

st.subheader("Module 4: boundaries and the Fourier domain")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("app_pages/m4_q1_rgb.py", label="Q1: human boundary in RGB photos", icon="🧍")
    st.caption("GrabCut and marker-based watershed, compared against SAM2.")
with c2:
    st.page_link("app_pages/m4_q2_thermal.py", label="Q2: human boundary in thermal images", icon="🌡️")
    st.caption("White top-hat and hysteresis thresholding on heat, compared against SAM2.")
with c3:
    st.page_link("app_pages/m4_q3_fourier.py", label="Q3: edges and regions in the Fourier domain", icon="〰️")
    st.caption("The derivations, plus a live lab: pick a filter H(u,v) and see what it keeps.")

st.subheader("Module 6: motion and structure")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("app_pages/m6_flow.py", label="Optical flow on video", icon="🌊")
    st.caption("Dense flow on a fixed camera and a moving one, frame by frame.")
with c2:
    st.page_link("app_pages/m6_tracking.py", label="Lucas–Kanade tracking, checked", icon="🎯")
    st.caption("The derived equations against OpenCV and measured pixel locations.")
with c3:
    st.page_link("app_pages/m6_sfm.py", label="Structure from motion of a book", icon="📐")
    st.caption("Four photos to a 3D cover and camera positions; try a wrong focal length.")

st.divider()
st.write("No deep learning or machine learning runs on these pages. SAM2 appears only "
         "as the reference: its masks were made in Meta's SAM2 demo and ship with the app.")
