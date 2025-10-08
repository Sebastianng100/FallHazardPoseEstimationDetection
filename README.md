*To run codes locally
PYTHON NEEDS TO BE IN 3.11.x else the mediapipe won't work
1. pip install -r requirements.txt
2. pip install -U gradio gradio_client
3. python src/app/gradio_app.py
4. Check browser at url:    http://127.0.0.1:7860 or http://0.0.0.0:7860
.venv\Scripts\activate

To run codes on docker

# Build image
go into parent folder first
docker build -t fall-detection-app .
# Run container
docker run -it --rm -p 7860:7860 fall-detection-app

<b>Docker depolyment</b>
project_root/
├── docker/
│   ├── app.Dockerfile
│   └── .dockerignore
├── saved_model/
│   ├── resnet_fall_model.pth
│   ├── efficientnet_fall_model.pth
├── processed_dataset/   # optional if you just need inference
├── testing.py           # Gradio app (frontend)
├── requirements.txt     # your dependencies

<b>Hugging face</b>
git lfs install

# Clone your empty Space
git clone https://huggingface.co/spaces/sngofficial100/fall-detection-demo
cd fall-detection-demo

# Copy your project files into this folder
cp -r ../your-local-project/* .

# Commit & push
git add .
git commit -m "Initial Docker deployment"
git push

<b>Downloaded dataset</b>
kaggle_fall_dataset
1. https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
kaggle_le2i dataset
2. https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia

Run Classification
1. python classification.py --epochs 10
2. python classification.py --epochs 10 --unfreeze-backbone


Questions for the prof
(Dataset)
1. Do i need to annotate some images via maybe labelmg to show i know how to do it
2. Do i need to use a good model that has high accuracy to spam label images and check through it to have a bigger dataset
(Model Archetecture)
1. Using resnet and yolo which is standard currently, try maybe efficientnet

Questions for myself
1. Have you done other model other than yolo :'(
2. modify layers
3. do cloud deploy
4. do annotation if have time


Struggles i faces
I forgot that my 0 is fall and 1 is not fall and i accidentally trained it opposite and was so frustrated as to why my results are so bad but actually it is just my fault for being clumsy

solo is too hard i need to do everything alone no help from anyone

cloud hosting was horror as lfs was tiring to get pass

# Model Overview

| Model                    | Architecture    | Backbone   | Optimisation                | Purpose                                                         |
| ------------------------ | --------------- | ---------- | --------------------------- | --------------------------------------------------------------- |
| ResNet Baseline          | ResNet-18       | Frozen     | Cross-Entropy               | Establish baseline accuracy using pre-trained ImageNet weights. |
| EfficientNet Baseline    | EfficientNet-B0 | Frozen     | Cross-Entropy               | Evaluate performance of a lightweight CNN.                      |
| ResNet (Optimised)       | ResNet-18       | Fine-tuned | Focal Loss, OneCycleLR      | Improve accuracy and handle class imbalance.                    |
| EfficientNet (Optimised) | EfficientNet-B0 | Fine-tuned | Label Smoothing, Mixup, SWA | Enhance generalisation and stability.                           |

All models classify between two classes:
0 = Fall, 1 = Not Fall.

# Model Modifications
| Model               | Original Output              | Modified Output       | Added                                |
| ------------------- | ---------------------------- | --------------------- | ------------------------------------ |
| **ResNet-18**       | 1000-class Linear(512→1000)  | Binary Linear(512→2)  | Dropout(0.5)                         |
| **EfficientNet-B0** | 1000-class Linear(1280→1000) | Binary Linear(1280→2) | Label smoothing, Mixup (in training) |


# Baseline vs Non-Baseline Models

<b>Baseline Models</b>

Baseline models were trained with frozen backbones and a single linear classifier layer. Only the final fully connected layer was updated, keeping ImageNet features fixed.
These models provide a reference for evaluating the effect of fine-tuning and regularisation.

<b>Non-Baseline Models</b>
The optimised models unfreeze the backbone for fine-tuning and apply advanced training strategies:
* <b>ResNet</b>: uses Focal Loss and OneCycleLR to focus learning on hard samples and improve convergence.

* <b>EfficientNet</b>: employs Label Smoothing, Mixup augmentation, and Stochastic Weight Averaging (SWA) for stability and calibration.

# Training Configuration

| Parameter         | Baseline Models | Non-Baseline Models                  |
| ----------------- | --------------- | ------------------------------------ |
| Epochs            | 10              | 20                                |
| Batch Size        | 16              | 16                                   |
| Optimizer         | Adam            | AdamW (ResNet) / Adam (EfficientNet) |
| Learning Rate     | 1e-4            | 1e-5 (with scheduler)                |
| Loss Function     | Cross-Entropy   | Focal Loss / Label Smoothing         |
| Scheduler         | None            | OneCycleLR / CosineAnnealingLR       |
| Backbone          | Frozen          | Fine-tuned                           |
| Augmentation      | None            | Mixup, Dropout, Label Smoothing      |
| Evaluation Metric | F1-Score        | F1-Score                             |

# Evaluation Results

| Model                    | Best Epoch | Best Validation F1 | Validation Accuracy | Observations                                                      |
| ------------------------ | ---------- | ------------------ | ------------------- | ----------------------------------------------------------------- |
| ResNet Baseline          | 8          | 0.75               | 0.82                | Stable but limited by frozen backbone.                            |
| EfficientNet Baseline    | 9          | 0.65               | 0.78                | Underfitting due to limited training.                             |
| ResNet (Optimised)       | 11         | **0.87**           | **0.89**            | Best performing model; well-balanced precision and recall.        |
| EfficientNet (Optimised) | 19         | 0.67               | 0.81                | Stable but slower convergence; mixup reduces short-term accuracy. |

# Key Findings

* Fine-tuning the backbone significantly improves performance compared to frozen baselines.
* Focal Loss effectively mitigates class imbalance in fall detection data.
* OneCycleLR accelerates convergence and avoids overfitting.
* Label Smoothing and SWA improve calibration but require longer training.
* ResNet (Optimised) achieved the best trade-off between accuracy, F1-score, and generalisation.

Running local pc:
run env
go to main folder
python main.py

Running local docker:
run env
go to root folder
docker run -it --rm -p 7860:7860 fall-detection-app

Running cloud huggingface:
