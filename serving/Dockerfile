# Servicio de inferencia (FastAPI) — detección YOLOv8m ONNX + identificación
# DINOv2 ViT-S/14 + cabeza afinada + galería. Inferencia en GPU si la hay.
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

# HOME=/app desde el inicio: el cache de pip/timm (~/.cache) queda bajo /app,
# que se le entrega a appuser mas abajo (si no, el cache quedaria en /root,
# illegible para el usuario sin privilegios y forzaria red en cada arranque).
ENV HOME=/app

# libgl1/libglib2.0-0: requeridos por opencv-python (dependencia de ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt ./serving/requirements.txt

# Variante de PyTorch. Por defecto cu128 (GPU local). Para Azure Container Apps,
# que solo tiene CPU, se construye con TORCH_VARIANT=cpu y ORT_PACKAGE=onnxruntime:
# la imagen pierde varios GB de CUDA y el arranque en frio se acorta.
ARG TORCH_VARIANT=cu128
ARG ORT_PACKAGE=onnxruntime-gpu

# torch/torchvision con CUDA 12.8. La imagen crece varios GB, pero DINOv2 a
# 518 px tarda del orden de un segundo por recorte en CPU y decenas de
# milisegundos en GPU: es lo que hace viable el analisis en vivo (RNF-04).
# Sin GPU visible, INFER_DEVICE resuelve a cpu y el servicio arranca igual.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install --no-cache-dir -r serving/requirements.txt

# onnxruntime-gpu ejecuta el detector ONNX en la GPU. Busca CUDA y cuDNN en el
# sistema, y aqui vienen dentro de los paquetes nvidia-* que arrastra torch,
# bajo site-packages/nvidia/*/lib.
RUN pip install --no-cache-dir ${ORT_PACKAGE}

ENV NVLIB=/usr/local/lib/python3.14/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVLIB}/cuda_runtime/lib:${NVLIB}/cudnn/lib:${NVLIB}/cublas/lib:${NVLIB}/cufft/lib:${NVLIB}/curand/lib

# Pre-descarga el backbone DINOv2 (timm) en build-time: evita que el primer
# request del contenedor dependa de salida a internet.
RUN python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)"

COPY src ./src
COPY serving ./serving
COPY models ./models
# Modo ensamble: pesos, galeria base y calibracion FUERA de /app/models. En Docker
# Compose esa carpeta es un volumen que solo se precarga la primera vez: unos pesos
# nuevos puestos ahi no aparecerian nunca en una maquina que ya lo tenia creado.
COPY models/ensamble ./ensamble
ENV ENSAMBLE_DIR=/app/ensamble

RUN useradd --uid 1000 --home-dir /app --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
