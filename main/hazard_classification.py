from ultralytics import YOLO
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_CONFIG = os.path.join(BASE_DIR, "hazard_dataset", "hazard.yaml")
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")  # where to save

def train_hazard_model():
    # Load YOLOv8n (nano) model - small and fast
    model = YOLO("yolov8n.pt")   # nano (best for CPU)  # or yolov8s.pt for more accuracy

    # Train
    '''model.train(
        data=DATA_CONFIG,
        epochs=30,
        imgsz=640,
        batch=16,
        workers=0,  # set 0 on Windows
        project=SAVE_DIR,   # 👈 tells YOLO where to save
        name="hazard_yolov8"   # 👈 subfolder inside saved_model/
    )'''

    model.train(
    data=DATA_CONFIG,
    epochs=15,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    verbose=False,
    project=SAVE_DIR,
    name="hazard_yolov8"
    )
    print(f"Training complete. Model saved in: {os.path.join(SAVE_DIR, 'hazard_yolov8', 'weights')}")

if __name__ == "__main__":
    train_hazard_model()
