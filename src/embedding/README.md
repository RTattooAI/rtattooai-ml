# Etapa de Identificación (embeddings + similitud coseno)

> **Estado: Paso 0 implementado (baseline sin entrenar).** Se usa **DINOv2**
> preentrenado como extractor de "huellas" y se compara contra una **galería**
> de tatuajes de banda por similitud coseno. No es un clasificador de
> "banda/no-banda": es **búsqueda por similitud + rechazo open-set**, con
> revisión humana (herramienta de apoyo preliminar).

## Paso 0 — baseline ejecutable

```bash
python -m src.embedding.step0_baseline      # requiere datasets/gang_tattoos/ y models/yolov8m_best.pt
```
- `embedder.py` — DINOv2 ViT-S/14 (384-d), L2-normalizado.
- `step0_baseline.py` — galería con los 25 tatuajes + variaciones 2D simuladas;
  mide recuperación (top-1/top-5), separación de identidades y rechazo open-set
  contra tatuajes no-banda recortados con el YOLO. Reporte en
  `reports/embedding_step0.md`.

**Resultado (sin entrenar):** top-1 ≈ 91%, open-set genuino ≈ 0.71 vs no-banda
≈ 0.28 (umbral ~0.49 separa bien). Confirma el enfoque antes de invertir en
datos sintéticos / fine-tuning.

> Aviso honesto: las "consultas" del Paso 0 son variaciones **2D sintéticas**
> de los mismos diseños; la validación real exige **fotos reales** (y/o renders
> Blender) — esa es la verdadera brecha sim-to-real pendiente.

---

## Diseño completo (siguiente)

## Qué irá aquí

La segunda etapa toma el recorte del tatuaje (bounding box producido por el
detector YOLOv8) y genera un **vector de embedding** con una CNN. La
identificación se hace por **similitud coseno** contra una base de embeddings
de tatuajes de bandas conocidas.

Flujo previsto:

```
imagen --> [YOLOv8 detección] --> recorte del tatuaje
       --> [CNN embeddings]   --> vector (p. ej. 512-d)
       --> similitud coseno contra galería --> top-K candidatos
```

## Estructura sugerida (futuro)

```
src/embedding/
├── model.py        # backbone CNN (p. ej. ResNet/EfficientNet) + cabeza de embedding
├── train.py        # entrenamiento métrico (triplet / ArcFace / contrastive)
├── build_gallery.py# extrae embeddings de la galería de referencia
├── identify.py     # similitud coseno + ranking top-K
└── export_onnx.py  # exporta el extractor de embeddings a ONNX
```

## Notas de diseño

- El extractor de embeddings también se exportará a ONNX para el backend .NET.
- Mantener la misma convención de `src/shared` (paths, seeds, métricas).
- Es una herramienta de **apoyo al análisis preliminar**, no de identificación
  definitiva: reportar siempre top-K con puntajes, no una única respuesta.
