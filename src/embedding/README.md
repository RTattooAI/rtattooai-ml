# Etapa de Identificación (CNN de embeddings) — PLACEHOLDER

> **Estado: no implementada todavía.** Esta carpeta queda preparada para la
> segunda etapa de IA del proyecto. La sesión actual se centra solo en la
> etapa de **detección (YOLOv8)**.

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
