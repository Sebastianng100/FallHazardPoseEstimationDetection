# Home.py
import streamlit as st

st.set_page_config(
    page_title="Fall Detection Yolov8m Dashboard",
    page_icon="",
    layout="wide",
)

st.title("Fall Detection Dashboard")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Project Overview")
    st.write(
        """
This dashboard showcases your **YOLOv8 fall & hazard detection model** (`train_advanced_v8m`).

Use the navigation menu on the left:

- **Image Upload**: Upload single or multiple images to visualize detections.
- **Video Demo**: Upload a video or use the webcam snapshot to see detection results.

The goal is to demonstrate:
- Reliable detection performance
- Practical, user-friendly UI
- Clear visual evidence for grading and real-world usability
        """
    )

with col2:
    st.subheader("Model Highlights")
    st.metric("Model Type", "YOLOv8m")
    st.metric("Task", "Fall Detection")
    st.metric("Status", "Trained & Deployed in App")

st.markdown("### How to use")
st.write(
    """
1. Go to **Image Upload** page to test on photos.
2. Go to **Video Demo** page to test on videos or webcam snapshots.
3. Adjust **confidence threshold** to control sensitivity (higher = fewer false positives).
"""
)
