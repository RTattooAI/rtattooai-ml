"""Demo interactiva del detector de tatuajes RTattooAI (Streamlit).

Permite probar el modelo YOLOv8 entrenado con:
  - una imagen subida, o
  - una foto tomada al momento con la cámara.

Ejecutar (desde la raíz del repo, con el venv activo):
    streamlit run app/streamlit_app.py

Es una herramienta de APOYO al análisis preliminar, no de identificación
definitiva: muestra cajas y confianzas, siempre con validación humana.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# permite importar src/ y resolver rutas del repo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="RTattooAI — Detección de tatuajes", page_icon="🔍",
                   layout="wide")


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def available_models() -> list[Path]:
    if MODELS_DIR.exists():
        return sorted(MODELS_DIR.glob("*_best.pt"))
    return []


def run_inference(model, image: Image.Image, conf: float, iou: float):
    """Corre el modelo sobre una imagen PIL y devuelve (anotada_RGB, detecciones).

    IMPORTANTE: se pasa la imagen PIL (RGB) directo a predict. Si se pasara un
    array numpy, Ultralytics lo interpretaría como BGR e invertiría los canales
    (la imagen saldría azulada y el modelo recibiría los colores cambiados).
    """
    image = image.convert("RGB")
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    res = results[0]
    annotated = res.plot()[:, :, ::-1]  # plot() devuelve BGR -> a RGB para mostrar
    dets = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                "confianza": round(float(b.conf[0]), 3),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            })
    dets.sort(key=lambda d: d["confianza"], reverse=True)
    return annotated, dets


# ----------------------------- UI -----------------------------
st.title("🔍 RTattooAI — Detección de tatuajes")
st.caption(
    "Demo de la etapa de **detección (YOLOv8)**. Herramienta de **apoyo al "
    "análisis preliminar**, no de identificación definitiva. Requiere "
    "validación humana."
)

models = available_models()
if not models:
    st.error(
        "No se encontró ningún modelo en `models/*_best.pt`.\n\n"
        "Entrena primero con `python -m src.detection.train` "
        "(o copia un `.pt` a la carpeta `models/`)."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuración")
    model_path = st.selectbox(
        "Modelo", models, format_func=lambda p: p.name,
        index=next((i for i, p in enumerate(models) if "yolov8m" in p.name), 0),
    )
    conf = st.slider("Confianza mínima", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.markdown(
        "**Cómo usar**\n\n"
        "1. Elige *Subir imagen* o *Tomar foto*.\n"
        "2. Ajusta la confianza si hay falsos positivos/negativos.\n"
        "3. Revisa las cajas y sus confianzas."
    )

model = load_model(str(model_path))

tab_subir, tab_camara = st.tabs(["📁 Subir imagen", "📷 Tomar foto"])

images_to_process: list[tuple[str, Image.Image]] = []

with tab_subir:
    files = st.file_uploader(
        "Sube una o varias imágenes", type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    for f in files or []:
        images_to_process.append((f.name, Image.open(f)))

with tab_camara:
    st.write("Toma una foto a un tatuaje con tu cámara:")
    shot = st.camera_input("Cámara")
    if shot is not None:
        images_to_process.append(("foto_camara.jpg", Image.open(shot)))

if not images_to_process:
    st.info("Sube una imagen o toma una foto para empezar.")
    st.stop()

for name, image in images_to_process:
    st.subheader(name)
    with st.spinner("Detectando…"):
        annotated, dets = run_inference(model, image, conf, iou)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(annotated, caption="Detecciones", use_container_width=True)
    with c2:
        st.metric("Tatuajes detectados", len(dets))
        if dets:
            st.dataframe(dets, use_container_width=True, hide_index=True)
            st.caption(
                f"Confianza máx: {dets[0]['confianza']:.2f} · "
                f"mín: {dets[-1]['confianza']:.2f}"
            )
        else:
            st.warning("No se detectaron tatuajes con este umbral. "
                       "Prueba bajar la *Confianza mínima*.")
    st.divider()
