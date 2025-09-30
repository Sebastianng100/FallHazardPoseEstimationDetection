*To run codes locally
PYTHON NEEDS TO BE IN 3.11.x else the mediapipe won't work
1. pip install -r requirements.txt
2. pip install -U gradio gradio_client
3. python src/app/gradio_app.py
4. Check browser at url:    http://127.0.0.1:7860 or http://0.0.0.0:7860
.venv\Scripts\activate

*To run codes on docker
1. docker build -f docker/app.Dockerfile -t fallguard:latest .
2. docker run -p 7860:7860 fallguard:latest

Downloaded dataset
1. https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia

Run Classification
1. python classification.py --epochs 10
2. python classification.py --epochs 10 --unfreeze-backbone


Questions for the prof
1. Do i need to annotate some images via maybe labelmg to show i know how to do it
2. Do i need to use a good model that has high accuracy to spam label images and check through it to have a bigger dataset

Questions for myself
1. Have you done other model other than yolo :'(