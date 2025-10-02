# Use official Python base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system deps for Pillow, etc.
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port for Gradio
EXPOSE 7860

# Run your Gradio app
CMD ["python", "testing.py"]
