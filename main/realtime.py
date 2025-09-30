import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------- Load Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("saved_model/fall_model.pth", map_location=device))
model.to(device)
model.eval()

CLASSES = ["Not Fall", "Fall"]

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- OpenCV Loop ----------
cap = cv2.VideoCapture(0)   # 0 = default webcam

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert frame to PIL and run inference
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    label = f"{CLASSES[pred.item()]} ({conf.item()*100:.1f}%)"

    # draw prediction on frame
    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Real-time Fall Detection", frame)

    # break loop when 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
