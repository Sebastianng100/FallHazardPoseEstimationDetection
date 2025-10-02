import gradio as gr
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Model paths
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "resnet_fall_model.pth")
EFFICIENTNET_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "efficientnet_fall_model.pth")
RESNET_BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "resnet_baseline_fall_model.pth")
EFFICIENTNET_BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "efficientnet_baseline_fall_model.pth")
HAZARD_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "hazard_yolov83", "weights", "best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fall_classes = ["Not Fall", "Fall"]

# ---------------- Load Models ----------------
# ResNet (non-baseline)
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(resnet_model.fc.in_features, len(fall_classes))
)
resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
resnet_model.to(device).eval()

# ResNet (baseline)
resnet_baseline_model = models.resnet18(weights=None)
resnet_baseline_model.fc = nn.Linear(resnet_baseline_model.fc.in_features, len(fall_classes))
resnet_baseline_model.load_state_dict(torch.load(RESNET_BASELINE_MODEL_PATH, map_location=device))
resnet_baseline_model.to(device).eval()

# EfficientNet (non-baseline)
efficientnet_model = models.efficientnet_b0(weights=None)
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, len(fall_classes))
efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=device))
efficientnet_model.to(device).eval()

# EfficientNet (baseline)
efficientnet_baseline_model = models.efficientnet_b0(weights=None)
efficientnet_baseline_model.classifier[1] = nn.Linear(efficientnet_baseline_model.classifier[1].in_features, len(fall_classes))
efficientnet_baseline_model.load_state_dict(torch.load(EFFICIENTNET_BASELINE_MODEL_PATH, map_location=device))
efficientnet_baseline_model.to(device).eval()

# Hazard Detection
hazard_model = YOLO(HAZARD_MODEL_PATH)

# ---------------- Transform ----------------
fall_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Prediction Helpers ----------------
def predict_model(image, model, model_name):
    img = Image.fromarray(image)
    img_t = fall_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]

    conf, pred = torch.max(probs, 0)
    label = fall_classes[pred.item()]

    probs_dict = {fall_classes[i]: f"{probs[i].item()*100:.1f}%" for i in range(len(fall_classes))}
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

    return (
        f"[{model_name}]\n"
        f"Prediction: {label}\n"
        f"Confidence: {conf.item()*100:.1f}%\n"
        f"Probabilities: {probs_dict}\n"
        f"Uncertainty (entropy): {entropy:.3f}"
    )

def predict_hazard(image):
    results = hazard_model.predict(image, conf=0.25)
    return results[0].plot()

# ---------------- Gradio Interface ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧍 Fall & Hazard Detection Demo")

    mode = gr.Dropdown(
        choices=[
            "ResNet Fall Detection (Baseline)",
            "ResNet Fall Detection (Non-Baseline)",
            "EfficientNet Fall Detection (Baseline)",
            "EfficientNet Fall Detection (Non-Baseline)",
            "Hazard Detection"
        ], 
        label="Select Mode", 
        value="ResNet Fall Detection (Baseline)"
    )

    img = gr.Image(type="numpy", label="Upload Image")
    run = gr.Button("▶ Run Inference", variant="primary")

    out_txt = gr.Textbox(label="Prediction Results", lines=5)
    out_img = gr.Image(type="numpy", label="Hazard Detection Output")

    def inference(image, mode):
        if mode == "ResNet Fall Detection (Baseline)":
            return predict_model(image, resnet_baseline_model, "ResNet Baseline"), None
        elif mode == "ResNet Fall Detection (Non-Baseline)":
            return predict_model(image, resnet_model, "ResNet Non-Baseline"), None
        elif mode == "EfficientNet Fall Detection (Baseline)":
            return predict_model(image, efficientnet_baseline_model, "EfficientNet Baseline"), None
        elif mode == "EfficientNet Fall Detection (Non-Baseline)":
            return predict_model(image, efficientnet_model, "EfficientNet Non-Baseline"), None
        else:
            return None, predict_hazard(image)

    run.click(inference, inputs=[img, mode], outputs=[out_txt, out_img])

if __name__ == "__main__":
    demo.launch(share=True)
