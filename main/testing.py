import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "fall_model.pth")

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Load Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# class names
CLASSES = ["Not Fall", "Fall"]

# Hardcode best validation metrics from your last training
VAL_METRICS = {
    "Precision": 0.805,   # from epoch 7-10 unfreeze run
    "Recall": 0.943,
    "F1-score": 0.868,
    "Validation Accuracy": 0.82
}

# ---------- Inference ----------
def predict_fall(image):
    if image is None:
        return "No Image Uploaded", "", ""

    img = Image.fromarray(image)
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    label = CLASSES[pred.item()]
    confidence = f"Confidence: {conf.item()*100:.1f}%"
    
    metrics_str = "\n".join([f"{k}: {v:.3f}" for k, v in VAL_METRICS.items()])
    return label, confidence, metrics_str


# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## 🧍 Fall Classification (Trained Model)")
    img = gr.Image(type="numpy", label="Upload Image")
    run = gr.Button("▶ Run Inference", variant="primary")
    out_txt = gr.Textbox(label="Prediction", lines=2)
    out_conf = gr.Textbox(label="Model Confidence", lines=2)
    out_metrics = gr.Textbox(label="Validation Metrics (from training)", lines=6)
    run.click(predict_fall, inputs=img, outputs=[out_txt, out_conf, out_metrics])

if __name__ == "__main__":
    demo.launch(share=True)
