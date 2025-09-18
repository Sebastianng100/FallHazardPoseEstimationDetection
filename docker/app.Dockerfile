FROM python:3.11-slim

# (optional but handy for video later)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 7860
# -u = unbuffered logs so you can see errors immediately
ENTRYPOINT ["python","-u","src/app/gradio_app.py"]
