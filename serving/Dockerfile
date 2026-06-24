# Servicio de inferencia (FastAPI) — detección YOLOv8m ONNX + identificación
# DINOv2 ViT-S/14 + cabeza afinada + galería. Inferencia en CPU.
#
# Build (contexto = raíz del repo, para poder copiar src/ y models/):
#   docker build -f serving/Dockerfile -t rtattooai-serving .
# Run:
#   docker run -p 8000:8000 --env-file .env rtattooai-serving
#
# Listo para Azure Container Apps: puerto configurable por $PORT, secretos
# (AZURE_STORAGE_CONNECTION_STRING, etc.) por variables de entorno.

FROM python:3.14-slim

WORKDIR /app

# libgl1/libglib2.0-0: requeridos por opencv-python (dependencia de ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt ./serving/requirements.txt

# torch/torchvision CPU-only (evita arrastrar binarios CUDA innecesarios en el contenedor)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r serving/requirements.txt

# Pre-descarga el backbone DINOv2 (timm) en build-time: evita que el primer
# request del contenedor dependa de salida a internet.
RUN python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)"

COPY src ./src
COPY serving ./serving
COPY models ./models

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
