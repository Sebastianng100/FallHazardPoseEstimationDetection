import io

import streamlit as st
from PIL import Image

from app_utils import load_model, run_inference_on_image

st.set_page_config(
    page_title="Image Upload - Fall Detection",
    page_icon="",
    layout="wide",
)

st.title("Image Upload Demo")
st.markdown("Upload one or more images to visualize model predictions.")

# Sidebar controls
with st.sidebar:
    st.header("⚙ Inference Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Detections below this confidence will be filtered out."
    )

    st.caption("Model will be loaded once and reused across all inferences.")

model = load_model()

uploaded_files = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least one image to begin.")
else:
    st.markdown("---")
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.warning(f"Skipping file `{uploaded_file.name}` (not a valid image).")
            continue

        st.subheader(f"Image {idx}: `{uploaded_file.name}`")

        col1, col2 = st.columns([1.2, 1])

        annotated_image, detections = run_inference_on_image(
            model,
            image,
            conf=conf_threshold
        )

        with col1:
            st.write("**Original vs Detected**")
            img_cols = st.columns(2)
            with img_cols[0]:
                st.image(image, caption="Original", use_column_width=True)
            with img_cols[1]:
                st.image(annotated_image, caption="Detections", use_column_width=True)

        with col2:
            st.write("**Detection Summary**")
            if len(detections) == 0:
                st.warning("No objects detected above the selected confidence threshold.")
            else:
                for cls_name, conf in detections:
                    st.write(f"- **{cls_name}** — {conf:.2f} confidence")

            buf = io.BytesIO()
            annotated_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Annotated Image",
                data=byte_im,
                file_name=f"annotated_{uploaded_file.name}",
                mime="image/png"
            )

        st.markdown("---")
