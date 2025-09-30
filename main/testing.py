import gradio as gr
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FALL_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "fall_model.pth")
HAZARD_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "hazard_yolov83", "weights", "best.pt")

# ---------- Load Models ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Fall Model Setup ----------
fall_classes = ["Not Fall", "Fall"]

# Rebuild ResNet18 architecture
fall_model = models.resnet18(weights=None)
fall_model.fc = nn.Linear(fall_model.fc.in_features, len(fall_classes))

# Load weights into model
fall_model.load_state_dict(torch.load(FALL_MODEL_PATH, map_location=device))
fall_model.to(device)
fall_model.eval()

fall_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Hazard Detection Setup ----------
hazard_model = YOLO(HAZARD_MODEL_PATH)

# ---------- Prediction Functions ----------
def predict_fall(image):
    img = Image.fromarray(image)
    img_t = fall_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = fall_model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    label = fall_classes[pred.item()]
    return f"{label} (Confidence {conf.item()*100:.1f}%)"

def predict_hazard(image):
    results = hazard_model.predict(image, conf=0.25)
    return results[0].plot()  # annotated image with boxes

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 🧍 Fall & Hazard Detection")

    mode = gr.Dropdown(choices=["Fall Detection", "Hazard Detection"], 
                       label="Select Mode", value="Fall Detection")

    img = gr.Image(type="numpy", label="Upload Image")
    run = gr.Button("▶ Run Inference", variant="primary")

    out_txt = gr.Textbox(label="Prediction", lines=2)
    out_img = gr.Image(type="numpy", label="Hazard Detection Output")

    def inference(image, mode):
        if mode == "Fall Detection":
            return predict_fall(image), None
        else:
            return None, predict_hazard(image)

    run.click(inference, inputs=[img, mode], outputs=[out_txt, out_img])

if __name__ == "__main__":
    demo.launch(share=True)
