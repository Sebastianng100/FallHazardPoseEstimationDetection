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