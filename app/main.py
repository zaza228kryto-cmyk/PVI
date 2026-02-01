import hashlib
import io

import streamlit as st
from PIL import Image

from pvi_core.inference import FINDINGS, predict_probs, grad_cam, score_cam


st.set_page_config(page_title="PVI — Pulmo Visual Insight", layout="centered")

st.title("PVI — Pulmo Visual Insight")
st.caption("MVP: EfficientNet-B0 inference + Grad-CAM / Score-CAM explanation.")

uploaded = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Sidebar controls
st.sidebar.header("Explanation")
cam_method = st.sidebar.radio("Method", options=["Grad-CAM (fast)", "Score-CAM (advanced, slower)"], index=0)
show_cam = st.sidebar.checkbox("Show CAM", value=True)
alpha = st.sidebar.slider("Overlay transparency (alpha)", min_value=0.0, max_value=0.8, value=0.30, step=0.05)
use_contours = st.sidebar.checkbox("Contours (no fill)", value=False)

if cam_method.startswith("Score-CAM"):
    st.sidebar.info("Score-CAM is slower but often looks cleaner. Results are cached per image/label.")

# helper: stable key for caching
def _hash_uploaded_file(upl) -> str:
    data = upl.getvalue()
    return hashlib.sha256(data).hexdigest()

@st.cache_data(show_spinner=False)
def _cached_cam(image_bytes: bytes, target_index: int, method: str, alpha: float, contours: bool):
    img = Image.open(io.BytesIO(image_bytes))
    if method.startswith("Score-CAM"):
        res = score_cam(img, target_index=target_index, alpha=float(alpha), max_channels=64)
    else:
        res = grad_cam(img, target_index=target_index, alpha=float(alpha))
    return res.contours if contours else res.overlay

if uploaded is None:
    st.info("Upload an image to continue.")
else:
    img = Image.open(uploaded)

    with st.spinner("Running inference..."):
        out = predict_probs(img)

    st.subheader("Findings (probabilities)")
    for k in FINDINGS:
        v = float(out.get(k, 0.0))
        st.write(f"{k}: {v * 100:.1f}%")

    st.subheader("Explanation")
    target_label = st.selectbox("Target label", options=FINDINGS, index=0)
    target_index = FINDINGS.index(target_label)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img.convert("RGB"), caption="Original", use_container_width=True)

    with col2:
        if show_cam:
            image_bytes = uploaded.getvalue()
            with st.spinner("Computing CAM..."):
                out_img = _cached_cam(
                    image_bytes=image_bytes,
                    target_index=int(target_index),
                    method=cam_method,
                    alpha=float(alpha),
                    contours=bool(use_contours),
                )
            cap = "CAM contours" if use_contours else "CAM overlay (soft)"
            st.image(out_img, caption=f"{cap} — {cam_method}", use_container_width=True)
        else:
            st.info("CAM is disabled (toggle in sidebar).")

    st.caption("Note: CAM shows model attention, not a segmentation mask.")
