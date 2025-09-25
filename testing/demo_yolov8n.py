"""
demo_yolov8n.py
Mock "completed project" pipeline for Fall & Hazard Detection.
- Uses YOLOv8n pretrained model (COCO weights)
- Provides fake fall + hazard logic for demonstration
- Front-end is Gradio (image/video input, run button, outputs)
- Replace placeholder logic later with your own trained models
"""

from ultralytics import YOLO
import gradio as gr
import cv2
import os

# -------------------------------
# 1. Load Pretrained YOLOv8n
# -------------------------------
yolo_model = YOLO("yolov8n.pt")  # small, fast, pretrained model

HAZARD_CLASSES = ["knife", "scissors", "bench", "chair", "stairs"]  # fake hazard list

# -------------------------------
# 2. Fake Inference Functions
# -------------------------------

def analyze_frame(image_path):
    """Run YOLOv8n on one image, apply fake fall + hazard rules"""
    results = yolo_model.predict(image_path, conf=0.5, verbose=False)

    fall_status = "Not Fall"
    hazards = []
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": cls_name,
                "conf": round(conf, 2),
                "bbox": (round(x1), round(y1), round(x2), round(y2))
            })

            # --- Fake Fall Rule ---
            if cls_name == "person":
                w = x2 - x1
                h = y2 - y1
                if w > h:  # bounding box wider than tall = lying down
                    fall_status = "Fall"

            # --- Fake Hazard Rule ---
            if cls_name in HAZARD_CLASSES:
                hazards.append(cls_name)

    return {
        "fall_status": fall_status,
        "hazards": hazards,
        "detections": detections
    }


def process_input(file):
    """
    Handle both images and videos:
    - If image → run analyze_frame directly
    - If video → sample first frame and run analyze_frame
    """
    if file is None:
        return "No file uploaded."

    filepath = file.name
    ext = os.path.splitext(filepath)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        result = analyze_frame(filepath)
    elif ext in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        if not ret:
            return "⚠️ Could not read video."
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        result = analyze_frame(temp_frame_path)
        cap.release()
    else:
        return f"⚠️ Unsupported file type: {ext}"

    return f"Fall: {result['fall_status']} | Hazards: {result['hazards']} | Detections: {result['detections']}"
    

# -------------------------------
# 3. Gradio Front-End
# -------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Fall & Hazard Detection Demo — YOLOv8n")

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["YOLOv8n Fusion Model (Demo Only)"],
            label="Select Model",
            value="YOLOv8n Fusion Model (Demo Only)"
        )

    with gr.Row():
        file_input = gr.File(
            label="Upload Image or Video",
            file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi"]
        )

    run_button = gr.Button("▶ Run Inference", variant="primary")
    output_box = gr.Textbox(label="Result", interactive=False)

    run_button.click(fn=process_input, inputs=[file_input], outputs=output_box)


# -------------------------------
# 4. Run App
# -------------------------------
if __name__ == "__main__":
    demo.launch(share=True)   # gets you a public link
