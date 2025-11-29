import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app_utils import (
    load_model,
    run_inference_on_image,
)

st.set_page_config(
    page_title="Real-Time Video Demo - Fall Detection",
    page_icon="🎥",
    layout="wide",
)

st.title("Real-Time Webcam Fall Detection")
st.markdown(
    "Live YOLO detection with continuously updated bounding boxes from your webcam."
)

model = load_model()

with st.sidebar:
    st.header("⚙ Inference Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Filter out detections below this confidence.",
    )

    st.markdown("---")
    run_webcam = st.checkbox("Start Webcam Detection", value=False)
    st.caption("Uncheck or refresh the page to stop.")

placeholder = st.empty()
fps_placeholder = st.empty()

if run_webcam:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access webcam. Make sure no other app is using it.")
    else:
        prev_time = 0.0

        while run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam.")
                break

            curr_time = cv2.getTickCount() / cv2.getTickFrequency()
            if curr_time - prev_time < 0.03:
                continue
            dt = curr_time - prev_time
            prev_time = curr_time

            results = model.predict(
                frame,
                conf=conf_threshold,
                verbose=False,
            )

            annotated_bgr = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # Show live frame with boxes
            placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

            fps = 1.0 / (dt + 1e-8)
            fps_placeholder.markdown(f"**FPS:** {fps:.2f}")
            run_webcam = st.session_state.get("Sidebar ▶ Start Webcam Detection", True)

        cap.release()

else:
    st.info("Turn on **'Start Webcam Detection'** in the sidebar to begin streaming.")
