# Test estadístico pareado — yolov8m vs yolov8s (split `test`)

- Imágenes comparadas (mismo set, ambos modelos): **701**.
- Operación: conf=0.25, IoU-NMS=0.45, IoU-match (TP)=0.5.
- Test: **Wilcoxon signed-rank** (2 muestras pareadas; Friedman no aplica con solo 2 modelos).

## Resumen por imagen
| Métrica | yolov8m (media ± sd) | yolov8s (media ± sd) |
| --- | ---: | ---: |
| F1 @ IoU≥0.5 | 0.9042 ± 0.2525 | 0.9116 ± 0.2416 |
| IoU medio (TPs) | 0.8250 ± 0.2270 | 0.8249 ± 0.2176 |

## Wilcoxon signed-rank
- **F1 por imagen**: W = 1017.0, p = 0.1315 → **sin diferencia significativa** (α=0.05)
- **IoU medio por imagen**: W = 97302.0, p = 0.0047 → diferencia significativa (α=0.05)
