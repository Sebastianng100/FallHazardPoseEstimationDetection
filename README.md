*To run codes locally
PYTHON NEEDS TO BE IN 3.11.x else the mediapipe won't work
1. pip install -r requirements.txt
2. pip install -U gradio gradio_client
3. python src/app/gradio_app.py
4. Check browser at url:    http://127.0.0.1:7860 or http://0.0.0.0:7860
.venv\Scripts\activate

To run codes on docker

# Build image
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