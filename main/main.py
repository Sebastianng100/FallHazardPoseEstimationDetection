import gradio as gr
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATHS = {
    "ResNet Fall Detection": os.path.join(BASE_DIR, "saved_model", "resnet_fall_model.pth"),
    "EfficientNet Fall Detection": os.path.join(BASE_DIR, "saved_model", "efficientnet_fall_model.pth"),
    "ResNet Baseline Fall Detection": os.path.join(BASE_DIR, "saved_model", "resnet_baseline_fall_model.pth"),
    "EfficientNet Baseline Fall Detection": os.path.join(BASE_DIR, "saved_model", "efficientnet_baseline_fall_model.pth"),
}

HAZARD_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "hazard_yolov83", "weights", "best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#fall_classes = ["Not Fall", "Fall"]
fall_classes = ["Fall", "Not Fall"]

def load_resnet(model_path):
    model = models.resnet18(weights=None)
    is_baseline = "baseline" in os.path.basename(model_path).lower()
    if is_baseline:
        model.fc = nn.Linear(model.fc.in_features, len(fall_classes))  # baseline head
    else:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(fall_classes))
        )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)  # now heads match
    model.to(device).eval()
    return model

def load_efficientnet(model_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(fall_classes))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model

resnet_model = load_resnet(MODEL_PATHS["ResNet Fall Detection"])
efficientnet_model = load_efficientnet(MODEL_PATHS["EfficientNet Fall Detection"])

resnet_baseline_model = load_resnet(MODEL_PATHS["ResNet Baseline Fall Detection"])
efficientnet_baseline_model = load_efficientnet(MODEL_PATHS["EfficientNet Baseline Fall Detection"])

hazard_model = YOLO(HAZARD_MODEL_PATH)

fall_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_fall(image, model, model_name):
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
        #f"Probabilities: {probs_dict}\n"
        #f"Uncertainty (entropy): {entropy:.3f}"
    )

def predict_hazard(image):
    results = hazard_model.predict(image, conf=0.25)
    return results[0].plot()

with gr.Blocks() as demo:
    gr.Markdown("Fall & Hazard Detection")

    mode = gr.Dropdown(
        choices=[
            "ResNet Fall Detection",
            "EfficientNet Fall Detection",
            "ResNet Baseline Fall Detection",
            "EfficientNet Baseline Fall Detection",
            #"Hazard Detection"
        ],
        label="Select Mode", 
        value="ResNet Fall Detection"
    )

    img = gr.Image(type="numpy", label="Upload Image")
    run = gr.Button("Run Inference", variant="primary")

    out_txt = gr.Textbox(label="Prediction", lines=4)
    out_img = gr.Image(type="numpy", label="Hazard Detection Output")

    def inference(image, mode):
        if mode == "ResNet Fall Detection":
            return predict_fall(image, resnet_model, "ResNet Fall Detection"), None
        elif mode == "EfficientNet Fall Detection":
            return predict_fall(image, efficientnet_model, "EfficientNet Fall Detection"), None
        elif mode == "ResNet Baseline Fall Detection":
            return predict_fall(image, resnet_baseline_model, "ResNet Baseline Fall Detection"), None
        elif mode == "EfficientNet Baseline Fall Detection":
            return predict_fall(image, efficientnet_baseline_model, "EfficientNet Baseline Fall Detection"), None
        else:
            return None, predict_hazard(image)

    run.click(inference, inputs=[img, mode], outputs=[out_txt, out_img])

if __name__ == "__main__":
    demo.launch(share=True)
