FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install CPU-only PyTorch first to avoid ~2GB CUDA wheels in container
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py face_detection_yunet_2023mar.onnx ./
COPY core ./core
COPY routes ./routes
COPY fas ./fas
COPY models ./models
COPY MobileFaceNet ./MobileFaceNet
COPY api_data ./api_data

EXPOSE 5000

CMD ["python", "app.py", "--source", "none", "--host", "0.0.0.0", "--port", "5000"]
