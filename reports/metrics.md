| modelo | split | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| yolov8m | val | 0.9237 | 0.7184 | 0.9311 | 0.8754 |
| yolov8m | test | 0.9149 | 0.681 | 0.8888 | 0.8657 |
| yolov8s | val | 0.9164 | 0.7008 | 0.8915 | 0.8803 |
| yolov8s | test | 0.9227 | 0.6809 | 0.8928 | 0.8726 |

## Comparación contra baseline y meta

- **Baseline** (Pocevičė et al., 2024, YOLOv5/DeMSI): mAP@0.5 = 0.79
- **Meta tesis**: mAP@0.5 > 0.85 y precisión >= 0.8

**Modelo definitivo `yolov8m` (split test):**

```
mAP@0.5 = 0.915  (CUMPLE meta > 0.85; supera baseline 0.79)
Precisión = 0.889  (CUMPLE meta >= 0.8)
```
