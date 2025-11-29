# app_utils.py
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO


# Change this if your model path is different
MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)

    # CORRECT WAY — modify the underlying model.names
    '''model.model.names = {
        0: "Fall",
        1: "Stand",
        2: "Stand",
        3: "Fall"
    }'''
    model.model.names = {
        0: "Like-Fall",
        1: "Fall",
        2: "Stand",
        3: "Stand"
    }

    return model



def run_inference_on_image(model, image: Image.Image, conf: float = 0.5):
    """
    Runs YOLO inference on a single PIL image and returns:
    - annotated_image (PIL)
    - detections list with (class_name, confidence)
    """
    img_rgb = np.array(image)
    results = model.predict(
        img_rgb,
        conf=conf,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()  # BGR numpy
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    detections = []
    names = model.model.names if hasattr(model.model, "names") else model.names

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            detections.append((cls_name, conf_score))

    return annotated_pil, detections


def save_uploaded_file_to_temp(uploaded_file):
    """
    Save an uploaded file to a temporary location and return the path.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return tmp_path


def process_video_file(model, input_path: str, output_path: str, conf: float = 0.5, frame_stride: int = 1):
    """
    Run YOLO inference on a video and save annotated video to output_path.
    frame_stride > 1 means processing every Nth frame (for speed).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_bar = st.progress(0, text="Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            # Run detection on this frame
            results = model.predict(
                frame,
                conf=conf,
                verbose=False
            )
            annotated = results[0].plot()
        else:
            annotated = frame

        out.write(annotated)

        frame_idx += 1
        if total_frames > 0:
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress, text=f"Processing video... ({int(progress * 100)}%)")

    cap.release()
    out.release()
    progress_bar.empty()
