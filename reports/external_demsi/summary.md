# Inferencia externa — demsi

> Reporte **cualitativo** (sin anotaciones de verdad → **no** es mAP).
> Modelo: `yolov8m_best.pt`, conf=0.25, iou=0.45.

- Imágenes procesadas: **1451**
- Imágenes con ≥1 detección: **1388 (95.7%)**
- Cajas detectadas en total: **2050**
- Confianza máx. por imagen (en las que detecta): media=0.852, mediana=0.897, mín=0.252, máx=0.974

*Nota:* sin etiquetas de referencia no se puede saber cuántas de estas detecciones son correctas (precisión) ni cuántos tatuajes se escaparon (recall). Para métricas reales se necesitan las anotaciones.
