| modelo | split | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| yolov8m | val | 0.5565 | 0.3896 | 0.8226 | 0.5174 |
| yolov8s | val | 0.5619 | 0.3944 | 0.7917 | 0.5371 |

## Comparación contra baseline y meta — dataset `demsi`

- **Baseline** (Pocevičė et al., 2024, YOLOv5/DeMSI): mAP@0.5 = 0.79
- **Meta tesis**: mAP@0.5 > 0.85 y precisión >= 0.8

**Modelo definitivo `yolov8m` (split val):**

```
mAP@0.5 = 0.556  (NO cumple meta > 0.85; no supera baseline 0.79)
Precisión = 0.823  (CUMPLE meta >= 0.8)
```
