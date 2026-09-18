# Multi-stage Dockerfile for FastAPI + PyTorch + OpenCV Deepfake Detection Backend
FROM python:3.10-slim AS base

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application source code and pre-bundled dataset samples
COPY backend/ ./backend/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY README.md .

# Create raw data storage directory
RUN mkdir -p data/raw data/processed models/checkpoints

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
