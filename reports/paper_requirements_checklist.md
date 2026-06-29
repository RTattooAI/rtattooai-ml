# Estado de "Requisitos para Paper" — checklist resuelto

> Mapea cada punto pendiente del paper a su estado actual. Generado a partir
> del código/logs del repo y de nuevos scripts de evaluación (ver "Fuente").

## Operacionales

| Punto | Estado | Detalle |
|---|---|---|
| GPU de entrenamiento | ✅ Resuelto | NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GB VRAM, local (no Colab/Azure). Fuente: `runs_train.log` / `runs_train_resume.log` (línea `GPU detectada: ...`). |
| vCPU/RAM del Container App de Azure | ✅ Resuelto (recomendación lista para desplegar) | **1 vCPU / 2 GiB**, plan Consumption, `min-replicas=0` (scale-to-zero) para quedarse dentro del grant gratis (180k vCPU-s + 360k GiB-s/mes, válido en cualquier cuenta incl. Azure for Students). Script listo: `serving/deploy/azure_deploy.sh` (ver `serving/deploy/README.md` para el detalle de costos — todo gratis salvo ACR Basic ~U$5/mes). Falta correrlo con la cuenta real del usuario para obtener la cifra de producción definitiva. |
| Latencia ms/img en producción | ⚠️ Medido localmente como proxy | `reports/serving_latency_local.md`: ~950-1300 ms/imagen (mediana), p95 ~2.5-2.8s, en CPU local (proxy de 1 vCPU con hilos limitados a 1). Pendiente re-medir contra el Container App real una vez corrido `azure_deploy.sh` (el script imprime el comando `curl` para hacerlo). |
| URL pública del repo | ❓ Pendiente — decisión del usuario | Remoto: `https://github.com/RTattooAI/rtattooai-ml.git`. La API de GitHub devuelve 404 sin auth → **actualmente privado**. Decidir si se hace público o se cita como "disponible bajo solicitud". |

## Dataset

| Punto | Estado | Detalle |
|---|---|---|
| Lista de fuentes públicas mezcladas | ✅ Resuelto | `reports/dataset_summary.md` + `reports/download_manifest.json`: 7 fuentes Roboflow (by_tatoo, code_with_heady, image_analysis_region, lance_nelson, playground_tattootest, pocevice_tattoo, tattoodet) + DeMSI (solo evaluación cross-domain, no entra al training). |
| Política de anonimización/cropping de personas | ❓ Pendiente — decisión del usuario | No existe ningún paso de blur/recorte de rostros en el pipeline; las fuentes Roboflow ya vienen pre-recortadas a la zona del tatuaje. Falta decidir cómo justificar esto en el paper. |
| Lista de data augmentations | ✅ Resuelto | `runs/detect/yolov8m/args.yaml`: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, flipud=0 (apagado a propósito), mosaic=1.0, auto_augment=randaugment, erasing=0.4; mixup/cutmix/copy_paste/shear/perspective/degrees=0. |

## Galería de identificación

| Punto | Estado | Detalle |
|---|---|---|
| Tamaño final de la galería | ✅ Resuelto | `models/gallery_dinov2.npz`: **25 identidades × 17 vistas (425 vectores) — confirmado como número DEFINITIVO** para la entrega. |
| Criterios de curación experta de simbología | ⚠️ Parcial | Sin documentación formal en el repo. Evidencia indirecta: el collage real de Tren de Aragua (`reports/real_photo_validation.md`) coincide con 5 diseños ya curados (estrella, corona, AK-47, reloj, locomotora) — sirve como cita, pero falta el criterio explícito/fuente experta para el resto de los 25. |

## Métricas

| Punto | Estado | Detalle |
|---|---|---|
| Top-5 y Top-10 | ✅ Calculado | `reports/identification_topk_cmc_far_frr.md`: Top-1 98.2%, Top-5 99.4%, Top-10 100.0% (galería de producción, consultas sintéticas, N=500). |
| mAP de retrieval | ✅ Calculado | mAP = 0.988 (mismo reporte). |
| FAR y FRR @ ρ | ✅ Calculado | A ρ=0.55 (operación actual): FAR ≈ 0.0%, FRR ≈ 11.8%. EER ≈ 11.1% en ρ≈0.32. |
| Valor numérico de ρ y margen | ✅ Resuelto | `src/embedding/match_config.json`: ρ=0.55, margen=0.05 (calibrado en Paso 0 sintético). |
| Curva CMC y sweep de ρ (Fig. 3) | ✅ Calculado | `reports/cmc_curve.png` y `reports/far_frr_sweep.png`. |
| Wilcoxon/Friedman yolov8m vs yolov8s | ✅ Calculado | `reports/stat_test_yolov8m_vs_yolov8s.md`: F1 por imagen sin diferencia significativa (p=0.13); IoU medio con diferencia estadísticamente significativa pero de magnitud despreciable (0.8250 vs 0.8249, p=0.0047). Friedman no aplica (solo 2 modelos) → se usó Wilcoxon, que es el test correcto. |

## Extra (no estaba en la lista original)

- `reports/real_photo_validation.md`: primera validación cualitativa con 6 fotos
  reales (no sintéticas), incluyendo un collage periodístico real de tatuajes
  de Tren de Aragua. Aporta evidencia real (no solo sintética) para la
  sección de identificación — sigue siendo N pequeño, no sustituye las ~20-30
  fotos reales etiquetadas que pide la validación definitiva.

## Caveat que aplica a TODO lo de identificación

Top-k/CMC/mAP/FAR-FRR usan la galería de producción real, pero consultas
**sintéticas** (variaciones 2D de los diseños limpios) — igual que el resto
de métricas de identificación del repo. La validación con fotos reales sigue
siendo el punto más importante pendiente para conclusiones definitivas.
