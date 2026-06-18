"""Demo del FLUJO COMPLETO de RTattooAI (detección + identificación).

  foto → [YOLO detecta y recorta] → [DINOv2 embedding] → coseno vs galería de
  bandas → top-K + veredicto (posible banda X / ninguna conocida).

Ejecutar (desde la raíz, con el venv activo):
    streamlit run app/streamlit_full.py

Herramienta de APOYO preliminar, no identificación definitiva. El umbral debe
calibrarse con fotos reales; los valores por defecto vienen del Paso 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shared import paths
from src.embedding.identify import TattooIdentifier, GALLERY_NPZ, DETECTOR

st.set_page_config(page_title="RTattooAI — Flujo completo", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Cargando detector + embeddings + galería…")
def load_identifier(threshold: float):
    return TattooIdentifier(threshold=threshold)


@st.cache_resource(show_spinner=False)
def gallery_thumbs():
    """Mapa nombre_limpio -> ruta de imagen del tatuaje de banda (para miniaturas)."""
    d = paths.DATASETS / "gang_tattoos"
    out = {}
    if d.exists():
        for p in d.glob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                out[p.stem.replace("-removebg-preview", "")] = p
    return out


def draw_boxes(image: Image.Image, results) -> Image.Image:
    im = image.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    w = max(2, im.size[0] // 250)
    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = r["box"]
        color = (40, 180, 99) if r["is_match"] else (210, 130, 40)
        dr.rectangle([x1, y1, x2, y2], outline=color, width=w)
        dr.text((x1 + 4, y1 + 4), f"#{i}", fill=color)
    return im


# ----------------------------- UI -----------------------------
st.title("🔎 RTattooAI — Flujo completo (detección → identificación)")
st.caption("Apoyo al análisis **preliminar**, no identificación definitiva. "
           "Todo resultado requiere validación humana.")

if not GALLERY_NPZ.exists() or not DETECTOR.exists():
    st.error(
        "Faltan artefactos:\n\n"
        f"- Detector: `{DETECTOR}` {'✅' if DETECTOR.exists() else '❌ (entrena/exporta YOLO)'}\n"
        f"- Galería: `{GALLERY_NPZ}` {'✅' if GALLERY_NPZ.exists() else '❌ (corre src.embedding.step0_baseline)'}"
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuración")
    conf = st.slider("Confianza detección (YOLO)", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
    threshold = st.slider("Umbral de coincidencia (coseno)", 0.1, 0.9, 0.45, 0.01,
                          help="Por encima = 'posible banda'. Calíbralo con fotos reales.")
    topk = st.slider("Top-K mostrados", 1, 8, 3)
    st.divider()
    st.markdown("**Flujo:** detectar → recortar → huella (DINOv2) → coseno vs galería.")

ident = load_identifier(threshold)
ident.threshold = threshold   # respeta el slider sin recargar el modelo
thumbs = gallery_thumbs()
st.caption(f"Galería: {len(ident.names)} tatuajes de banda.")

tab_subir, tab_cam = st.tabs(["📁 Subir imagen", "📷 Tomar foto"])
image = None
with tab_subir:
    f = st.file_uploader("Sube una foto", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if f:
        image = Image.open(f)
with tab_cam:
    shot = st.camera_input("Cámara")
    if shot:
        image = Image.open(shot)

if image is None:
    st.info("Sube una foto o toma una con la cámara para empezar.")
    st.stop()

with st.spinner("Detectando e identificando…"):
    results = ident.identify(image, conf=conf, iou=iou, topk=topk)

col_img, col_res = st.columns([3, 4])
with col_img:
    st.image(draw_boxes(image, results), caption="Detección", use_container_width=True)
    st.metric("Tatuajes detectados", len(results))

with col_res:
    if not results:
        st.warning("No se detectaron tatuajes. Prueba bajar la confianza de detección.")
    for i, r in enumerate(results, 1):
        top_name, top_score = r["matches"][0]
        if r["is_match"]:
            st.success(f"**Tatuaje #{i} → posible: {top_name}**  (coseno {top_score:.2f})")
        else:
            st.info(f"**Tatuaje #{i} → ninguna banda conocida**  (máx {top_score:.2f} < umbral)")
        c_crop, c_match = st.columns([1, 2])
        with c_crop:
            st.image(r["crop"], caption=f"recorte #{i}", use_container_width=True)
        with c_match:
            for name, score in r["matches"]:
                cols = st.columns([1, 3])
                tp = thumbs.get(name)
                if tp:
                    cols[0].image(str(tp), use_container_width=True)
                cols[1].write(f"{name}")
                cols[1].progress(min(max(score, 0.0), 1.0), text=f"coseno {score:.3f}")
        st.divider()

st.caption("⚠️ El veredicto es una **sugerencia por similitud**, no una identificación "
           "definitiva. Úsese solo como apoyo, con revisión de un analista.")
