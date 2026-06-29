# Métricas extendidas de identificación — Top-k, CMC, mAP, FAR/FRR

> Galería de PRODUCCIÓN actual (`models/gallery_dinov2.npz`): **25** identidades × ~17 vistas, embedding DINOv2 + cabeza afinada.
> Consultas: variaciones sintéticas 2D por diseño (rotación/color/fondo/ruido) — **orientativo**, no sustituye la validación con fotos reales (ver `reports/real_photo_validation.md`).

## Top-k y mAP de retrieval
- Consultas evaluadas: **500** (20 por diseño × 25 diseños).
- **Top-1: 98.2%** | **Top-5: 99.4%** | **Top-10: 100.0%**
- **mAP de retrieval: 0.988** (AP por consulta = 1/rank de la identidad correcta).

## Curva CMC (rank-k cumulative match)
| k | tasa acumulada |
| ---: | ---: |
| 1 | 98.2% |
| 2 | 99.2% |
| 3 | 99.2% |
| 5 | 99.4% |
| 10 | 100.0% |
| 15 | 100.0% |
| 20 | 100.0% |
| 25 | 100.0% |

## FAR / FRR en función del umbral ρ (margen fijo = 0.05)
- Punto de operación actual: **ρ = 0.55** → FAR ≈ **0.0%**, FRR ≈ **11.8%**.
- EER (equal error rate): **11.1%** en ρ ≈ **0.32**.

| ρ | FAR | FRR |
| ---: | ---: | ---: |
| 0.30 | 14.0% | 11.6% |
| 0.35 | 8.0% | 11.6% |
| 0.40 | 4.0% | 11.8% |
| 0.45 | 2.7% | 11.8% |
| 0.50 | 0.0% | 11.8% |
| 0.55 | 0.0% | 11.8% |
| 0.60 | 0.0% | 11.8% |
| 0.65 | 0.0% | 12.8% |
| 0.70 | 0.0% | 14.2% |
| 0.75 | 0.0% | 15.2% |
| 0.80 | 0.0% | 18.2% |
| 0.85 | 0.0% | 24.8% |
| 0.90 | 0.0% | 37.6% |
| 0.95 | 0.0% | 67.6% |
