# To run codes locally
PYTHON NEEDS TO BE IN 3.11.x else the mediapipe won't work
1. activate env .venv\Scripts\activate
2. pip install -r requirements.txt
3. pip install -U gradio gradio_client
4. python src/app/gradio_app.py
5. Check browser at url:    http://127.0.0.1:7860 or http://0.0.0.0:7860


# To run codes on docker
<b>Build image</b>
go into parent folder first
docker build -t fall-detection-app .
<b>Run container</b>
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

# Struggles i faced
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

# Model Fine tuning
| **Model**                       | **Fine-Tuned Components** | **Technique / Setting**               | **Purpose / Rationale**                                                                          |
| ------------------------------- | ------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **ResNet-18 (Optimised)**       | Backbone (Conv1–Layer4)   | **Unfrozen (trainable)**              | Allow full model to relearn posture-specific and domain-specific features for fall detection.    |
|                                 | Loss Function             | **Focal Loss**                        | Focuses learning on hard and minority samples (e.g., fall cases) to reduce class imbalance bias. |
|                                 | Learning Rate             | **1 × 10⁻⁵**                          | Small LR ensures slow, stable updates to pretrained weights during fine-tuning.                  |
|                                 | Epochs                    | **20**                                | Provides enough iterations for full-network convergence.                                         |
|                                 | Scheduler                 | **OneCycleLR**                        | Dynamically adjusts LR (increase then decrease) for faster, smoother convergence.                |
|                                 | Regularisation            | **Dropout (0.5)**                     | Prevents overfitting when all layers are trainable.                                              |
|                                 | Optimiser                 | **AdamW**                             | Stable weight-decay version of Adam; balances speed and generalisation.                          |
| **EfficientNet-B0 (Optimised)** | Backbone (All layers)     | **Unfrozen (trainable)**              | Fine-tune EfficientNet features to adapt to fall-specific visual cues.                           |
|                                 | Loss Function             | **Label Smoothing (0.1)**             | Prevents over-confidence and improves calibration on small datasets.                             |
|                                 | Data Augmentation         | **Mixup (α = 0.4)**                   | Blends images and labels to increase dataset diversity and robustness.                           |
|                                 | Optimisation              | **SWA (Stochastic Weight Averaging)** | Averages weights over epochs → smoother, more stable final model.                                |
|                                 | Scheduler                 | **CosineAnnealingLR**                 | Gradually decays LR following a cosine curve for stable long-term convergence.                   |
|                                 | Learning Rate / Epochs    | **1 × 10⁻⁵ / 20 epochs**              | Matches fine-tuning pace; stable for deep model adaptation.                                      |



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

# Training Rationale

| **Configuration Component**        | **Rationale / Purpose**                                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    **Baseline Configuration**      | **Goal:** Establish a simple, stable benchmark using pretrained ImageNet features.                                                                      |
| Model Backbone                     | Frozen. Reuses ImageNet filters (edges, shapes, textures) without further training. Prevents overfitting and provides a reliable reference performance. |
| Classifier Layer                   | Replaced final FC layer (1000→2) to match binary classes: *Fall* and *Not Fall.*                                                                        |
| Loss Function – `CrossEntropyLoss` | Standard classification loss measuring the difference between predicted and true classes. Easy to interpret and suitable for balanced datasets.         |
| Optimizer – `Adam`                 | Adaptive optimizer for quick convergence with minimal tuning. Performs well for small datasets and limited trainable parameters.                        |
| Learning Rate – `1e-4`             | Moderate LR ensures stable updates for the newly added classifier layer.                                                                                |
| Epochs – `10`                      | Sufficient for convergence when training only the final layer.                                                                                          |
| Data Augmentation                  | Basic resizing and ImageNet normalization to align with pretrained model expectations.                                                                  |
| **Outcome:**                       | Provides baseline metrics (accuracy, F1) for comparing improvement after full fine-tuning.                                                              |

| **Configuration Component**           | **Rationale / Purpose**                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
|   **Optimised ResNet Configuration** | **Goal:** Fine-tune all layers to learn posture-specific and domain-specific features for fall detection.                       |
| Backbone                              | Unfrozen. Allows convolutional filters to relearn human postures, orientations, and environmental cues specific to fall images. |
| Loss Function – `Focal Loss`          | Focuses learning on hard or minority samples (*Fall*), reducing class imbalance bias.                                           |
| Scheduler – `OneCycleLR`              | Dynamically adjusts learning rate (increase then decrease) for faster, smoother convergence. Prevents early stagnation.         |
| Regularisation – `Dropout(0.5)`       | Randomly disables 50% of neurons to reduce overfitting while fine-tuning all layers.                                            |
| Learning Rate – `1e-5`                | Smaller LR ensures slow, controlled fine-tuning of pretrained weights to prevent catastrophic forgetting.                       |
| Epochs – `20`                         | Provides sufficient updates for full-network fine-tuning and stability.                                                         |
| **Outcome:**                          | Achieved significant improvement in F1 (~0.87). Learns fall-specific posture and ground contact patterns.                       |

| **Configuration Component**                        | **Rationale / Purpose**                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
|   **Optimised EfficientNet Configuration**        | **Goal:** Enhance generalisation and stability through smoother training and better regularisation.                  |
| Backbone                                           | Unfrozen for complete fine-tuning; adapts complex EfficientNet features to fall imagery.                             |
| Loss Function – `Label Smoothing`                  | Prevents overconfidence by slightly softening label targets, improving calibration.                                  |
| Data Augmentation – `Mixup`                        | Blends two images and labels to create synthetic variations, improving generalisation and robustness.                |
| Optimisation – `SWA (Stochastic Weight Averaging)` | Averages model weights across epochs for smoother final performance and reduced variance.                            |
| Scheduler – `CosineAnnealingLR`                    | Gradually lowers learning rate following a cosine curve, improving long-term convergence.                            |
| Learning Rate – `1e-5`, Epochs – `20`              | Matches fine-tuning pace of ResNet while maintaining stability for complex layers.                                   |
| **Outcome:**                                       | Stable training, reduced overfitting, better calibration, and improved generalisation under limited data conditions. |

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

# Running
Running local pc:
run env
go to main folder
python main.py

Running local docker:
run env
go to root folder
docker run -it --rm -p 7860:7860 fall-detection-app

Running cloud huggingface:
git add .
git commit -m "Initial Docker deployment"
git push

# Project Part 2
## Why did i choose bounding boxes
Fall detection is fundamentally a spatial problem
Falls happen when a person’s body orientation and position change rapidly or assume an abnormal posture (e.g., lying on the ground).
Bounding boxes help to do the following:
1. locate the person
2. track the center of mass
3. detect posture shape
4. differentiate standing vs falling
Segmentation or Keypoint models such as pose is too slow and require GPU. YOLO bounding boxes gives me the compromise of:
1. speed
2. accuracy
3. simplicity
YOLO works very well with limited training data
LE2i dataset is small, noisy, and contains many camera angles.
YOLOv8 and etc. is designed for this kind of situation — fast learning, robust generalization.
## What i have attempted in this project half
Did my own annotation using roboflow as the cleanest dataset does not contain bounding boxes
The following was performed:
1. Reading annotation files
2. Generating YOLO label files automatically
3. Organizing train/val/test split
4. Creating data.yaml with class mapping
As now i am doing bounding box not classification, i cannot just use images with no bounding boxes
Originally Le2i has multiple classes such as falling lying like fall none sitting etc. but i decide to simplify it into 2 class standing and falling in the end to help:
1. Improves model accuracy
2. Reduces confusion between irrelevant states
3. Simplifies output for real-world usage
## Why I Used Light Augmentation and Disabled Mosaic / Mixup
The LE2i dataset is:
1. small
2. indoor only
3. low resolution
4. contains real human falls
5. Aggressive augmentations (mosaic, mixup) destroy human posture structure, making the fall pattern unrealistic.
So I disabled them.
## I HAD many models that i trained previously
### autoannotation-model(yolov8n) (Automatic Label Generation Test)
Why this model was created:
To test auto-annotation and see if YOLO could help label unlabelled images.
Settings Used:
1. Model: YOLOv8n
2. Epochs: 30 (just for inference)
No training purpose — used for auto-labeling
Outcome:
Auto-labeling produced inaccurate boxes
Needed manual correction
Useful for expanding dataset but not reliable alone
### Baseline model (Yolov8m - Default Settings)
I wanted to establish a starting point to see how well the model can perform with just the dataset and nothing futher this is so that when i do my changes i can see the grown in my model
Settings:
1. Model: yolov8m (Balanced model for testing)
2. Epochs: 10–20 (Avoid long training)
3. Batch size: 8 (Stable)
4. No augmentation tuning (Determine natural behaviour)
5. Default optimizer (SGD - Standard baseline)
6. Mosaic on (default YOLO behavior)
7. Mixup on

### Improved Model 1 (YOLOv8n - Fast)
This is my first point of improvement i wanted to look for a faster model and slightly more training
Settings:
1. Model: yolov8n (Fast and good for rapid iterations)
2. Epochs: 30 (More for learning patterns)
3. Batch size: 16 (Faster training, more stable gradiant estimate - Not too much to the point where generalization is bad and overfitting.)
4. Optimizer: AdamW (Smoother gradients and better for small dataset learning)
5. Mosaic = 0 (Fall posture distorted, disable)
6. Mixup = 0 (Human pose mixing unrealistic)

1. Baseline Model (YOLOv8m — Default Settings)
Why this model was created:
To establish a true baseline using default YOLO settings, no tuning.
This helps compare how each improvement affects accuracy.
Settings Used:
1. Model: YOLOv8m (balanced power + speed)
2. Epochs: 10–20 (quick baseline)
3. Batch size: 8 (safe for CPU)
4. Augmentations: default
5. Optimizer: SGD
6. Mosaic: ON
7. Mixup: ON
Outcome:
Moderate accuracy
Lots of false positives & false negatives
Useful only as a learning reference

### train7-yolov8n-baseline (Fast Baseline Model)
Why this model was created
To test if the tiny model (YOLOv8n) can still learn fall vs stand,
and to reduce training time dramatically.
Settings Used:
1. Model: YOLOv8n (fast, light)
2. Epochs: 30
3. Batch size: 16 (faster training)
4. Optimizer: AdamW
5. Mosaic: 0 (postures distort)
6. Mixup: 0
7. Light augmentations only
 - hsv
 - slight translate
 - horizontal flip
Outcome:
Faster than baseline
Better early learning stability
Still weaker detection (model too small)

### train8-yolov10n-baseline (Testing Next-Gen Architecture)
Why this model was created:
To check if YOLOv10n new architecture performs better than YOLOv8n.
Settings Used:
1. Model: YOLOv10n
2. Epochs: 60
3. Batch size: 16
4. Optimizer: AdamW
Augmentations: light only
No mosaic, no mixup
Outcome:
Slight improvement over v8n
Faster but not significantly better
Still not strong enough for fall posture detection

### autoannotation-model (Automatic Label Generation Test)
Why this model was created:
To test auto-annotation and see if YOLO could help label unlabelled images.
Settings Used:
1. Model: YOLOv8n
2. Epochs: 30 (just for inference)
No training purpose — used for auto-labeling
Outcome:
Auto-labeling produced inaccurate boxes
Needed manual correction
Useful for expanding dataset but not reliable alone

### yolo_v8m
Why this model was created:
To properly optimize YOLOv8m with carefully selected augmentations
and the AdamW optimizer for better learning.
Settings Used:
1. Model: YOLOv8m
2. Epochs: 150
3. Optimizer: AdamW
4. reduces overfitting
5. smooth learning
6. Batch size: 8
7. Augmentations:
- hsv adjustments
- translate
- scale
- flip
8. Mosaic: 0, Mixup: 0
(posture consistency required)
Outcome:
1. Big jump in accuracy
2. More stable bounding boxes

### yolo_v8m_v2 (Deep Augmentation)
Why this model was created
To test more aggressive augmentation and longer training.
Settings Used:
1. Model: YOLOv8m
2. Epochs: 150
3. Heavy Augmentations (controlled):
- more hsv
- more translate
- more scaling
- Optimizer: AdamW
- Batch size: 8
Outcome:
1. Better generalization
2. Lower validation error
3. But training time increased significantly
4. Slightly unstable because augmentations too strong

### yolov8m_refined
Why this model was created:
This is meant to be the final, polished, carefully tuned YOLOv8m configuration. However in the end YOLOV8m2 did the best.
Goal: maximum accuracy without extremely long training.
Settings Used:
1. Model: YOLOv8m
2. Epochs: 30
3. You stopped early due to time
4. YOLO Early stopping saved best weights
5. Batch size: 8
6. Optimizer: AdamW
7. Dropout: 0.1
8. Helps reduce overfitting
9. augmentations (balanced):
- hsv
- small translation
- small scaling
- flip left/right
10. mosaic = 0, mixup = 0
Outcome:
1. Highest accuracy out of all experiments
2. Good precision & recall
3. Stable and consistent bounding boxes
4. Minimal overfitting
5. Best real-world performance in your Streamlit app
