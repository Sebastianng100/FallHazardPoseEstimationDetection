from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_CONFIG = os.path.join(BASE_DIR, "hazard_dataset", "hazard.yaml")
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

def train_hazard_model():
    model = YOLO("yolov8n.pt")

    '''model.train(
        data=DATA_CONFIG,
        epochs=30,
        imgsz=640,
        batch=16,
        workers=0,
        project=SAVE_DIR,
        name="hazard_yolov8"
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
    print(f"Training complete.")

if __name__ == "__main__":
    train_hazard_model()
