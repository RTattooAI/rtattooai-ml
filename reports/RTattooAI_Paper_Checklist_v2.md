# RTattooAI — Checklist de requisitos para el paper

**Leyenda de estado**
- ✅ **Resuelto** — dato real ya disponible en el repo/logs.
- ⚙️ **Comprometido** — se ejecutará; método claro, resultado pendiente de medir.
- ❓ **Decisión pendiente** — requiere elección tuya antes de poder cerrarlo.
- ⚠️ **Parcial** — evidencia existe pero necesita complemento formal.

---

## 1. Entorno experimental

| Punto | Estado | Detalle y acción |
|---|---|---|
| GPU de entrenamiento | ✅ | NVIDIA RTX 3070 Ti Laptop GPU, 8 GB VRAM. Entrenamiento local en una sola GPU (no cloud). Fuente: `runs_train.log`. Reportar como "single-GPU local workstation" en la tabla de configuración. |
| vCPU / RAM de inferencia (producción) | ✅ | **Desplegado y verificado en Azure real** (29/06/2026): Azure Container Apps, plan Consumption, **1 vCPU / 2 GiB**, `min-replicas=0` (scale-to-zero), región East US 2. Arquitectura completa: 3 Container Apps (`rtattooai-serving` interno, `rtattooai-backend` y `rtattooai-frontend` externos) + Azure SQL Database serverless (Gen5, auto-pause 60 min, autenticación **Azure AD-only sin contraseña**) + Azure Storage Account (Blob). Costo real observado: ACR Basic ≈ US$5/mes; SQL serverless con auto-pause y Container Apps en Consumption se mantienen dentro del grant de Azure for Students para tráfico de demo/tesis. |
| Latencia ms/imagen (producción) | ✅ | **Medido sobre el Container App real** (1 vCPU/2GiB, CPU-only, n=10 peticiones `POST /identify` vía pipeline completo backend→IA): **mediana 2.51 s, media 2.55 s, mín 2.41 s, máx 3.00 s** (tiempo reportado por el propio servicio IA, `processingMs`). Extremo a extremo (incluye red + backend + Blob Storage): mediana 3.46 s, p95 (n=10) 3.95 s. **Cold start** (réplica en 0, primera petición tras inactividad): ~30 s — relevante reportarlo como característica de scale-to-zero, no como latencia de inferencia; se mitigó subiendo el timeout HTTP del backend de 30s a 100s para que el primer request no falle. Comparar con el proxy local (CPU con hilos limitados, mediana ~950–1300 ms) y explicar la diferencia por la CPU compartida del plan Consumption vs. la CPU dedicada del workstation local. |
| URL pública del repositorio | ❓ | Repo actualmente privado. Opciones: (a) hacerlo público + snapshot en Zenodo con DOI (recomendado para reproducibilidad), o (b) "disponible bajo solicitud". Antes de abrir, verificar historial de git en busca de secretos o imágenes de evidencia real. El dataset **no se redistribuye**: se incluye el `download_manifest.json` con fuentes y licencias. |

---

## 2. Dataset de detección

| Punto | Estado | Detalle y acción |
|---|---|---|
| Fuentes del dataset de entrenamiento | ✅ | 7 colecciones públicas de Roboflow (by_tatoo, code_with_heady, image_analysis_region, lance_nelson, playground_tattootest, pocevice_tattoo, tattoodet). DeMSI **solo para evaluación cross-domain**, no entra al training. Documentado en `reports/dataset_summary.md` y `reports/download_manifest.json`. Presentar como tabla con fuente, licencia y nº de imágenes en el paper. |
| Estadísticas del dataset consolidado | ✅ | 7 006 imágenes únicas / 8 619 cajas tras deduplicación. Partición 80/10/10 (train/val/test). Documentar el paso de deduplicación (hash perceptual) en la subsección de preprocesamiento. |
| Data augmentations aplicadas | ✅ | `runs/detect/yolov8m/args.yaml`: HSV (h=0.015, s=0.7, v=0.4), translate=0.1, scale=0.5, fliplr=0.5, **flipud=0** (apagado deliberadamente: el tatuaje tiene orientación canónica), mosaic=1.0, randaugment, erasing=0.4. mixup/cutmix/copy_paste/shear/perspective/degrees=0. Justificar en el paper las decisiones no-default. |
| Política de privacidad y anonimización | ⚙️ | Las fuentes Roboflow ya vienen recortadas a la zona del tatuaje. **Verificar** en una muestra representativa que no haya rostros identificables; si los hay, agregar paso de blur. Redactar subsección de consideraciones éticas que afirme: (1) datos recortados a la región del tatuaje, (2) el sistema no realiza reconocimiento facial, (3) las imágenes de evidencia operativa se tratan como datos sensibles (cifrado en reposo AES-256 en Azure Blob, HTTPS forzado, control de acceso por rol). Consultar con el asesor si UPC exige aval ético formal. |
| Evaluación cross-domain (DeMSI) | ✅ | mAP@0.5 zero-shot = 0.557. Documentar como "brecha de dominio honesta" en la sección de resultados; no confundir con las métricas de entrenamiento. Justifica la línea futura de reentrenamiento con imágenes peruanas. |

---

## 3. Galería de identificación

| Punto | Estado | Detalle y acción |
|---|---|---|
| Tamaño de la galería operativa | ✅ | **25 identidades × 17 vistas = 425 vectores** (confirmado como número definitivo). `models/gallery_dinov2.npz`, vectores de 128 dim. Presentar como galería extensible en runtime vía `/enroll` (el tamaño crece con el uso operativo). |
| Generación de vistas sintéticas | ✅ | Pipeline Blender: textura UV sobre modelos 3D, mínimo 5 variaciones de iluminación y 8 ángulos de cámara por tatuaje. Salida: PNG + anotaciones YOLO. Documentar el pipeline como contribución metodológica (permite escalar sin fotografías adicionales). |
| Criterios de curación de simbología | ⚠️ | Evidencia parcial: collage periodístico real de Tren de Aragua en `reports/real_photo_validation.md` corrobora 5 diseños (estrella, corona, AK-47, reloj, locomotora). **Acción necesaria:** redactar una subsección que documente explícitamente los criterios de selección de los 25 diseños, citando fuentes criminológicas/periodísticas y, de ser posible, validación de un experto PNP/DIRINCRI. Si la validación fue informal, declararlo como limitación. Sin esto, la afirmación de "identificación de bandas" queda sin respaldo formal. |
| Crecimiento del catálogo en producción | ⚙️ | El endpoint `/enroll` permite agregar tatuajes sin redesplegar. El sistema registra `addedBy`, `addedAt`, `source` y flag `validated` por entrada. **Se documentará** en el paper como un aporte operativo: el catálogo no es estático sino colaborativo, con trazabilidad de procedencia. |

---

## 4. Arquitectura del sistema

| Punto | Estado | Detalle y acción |
|---|---|---|
| Pipeline de detección (Etapa 1) | ✅ | YOLOv8m, entrada 640×640 (NCHW, RGB 0–1), salida [1,5,8400], NMS (conf≥0.25, IoU≥0.45). Exportado a ONNX (opset 12) para portabilidad. |
| Pipeline de identificación (Etapa 2) | ✅ | DINOv2 ViT-S/14 como extractor + cabeza contrastiva afinada (384→256→128 dim). Similitud coseno contra galería. Rechazo open-set: umbral ρ=0.55, margen=0.05. Documentar el rechazo open-set como diseño deliberado para herramienta de apoyo (evita falsos positivos de certeza). |
| Arquitectura web del sistema | ⚙️ | Tres capas: microservicio FastAPI (inferencia, interno), backend ASP.NET Core (.NET 10, orquestador), frontend Angular 21. Despliegue en Azure Container Apps (IA) + App Service (backend) + Static Web Apps (frontend) + SQL Database + Blob Storage. **Se documentará** con diagrama de arquitectura en el paper (ya comprometido en Hito 11 del Charter). |
| Validación humana en el flujo | ⚙️ | El sistema implementa human-in-the-loop: el analista puede marcar cada coincidencia como "validada / en revisión / descartada", con registro de quién y cuándo. El disclaimer es permanente e inamovible en la UI. **Se documentará** como mecanismo de salvaguarda ética, no como limitación. |

---

## 5. Métricas de detección

| Punto | Estado | Detalle y acción |
|---|---|---|
| mAP@0.5 y mAP@0.5:0.95 | ✅ | yolov8m: mAP@0.5=0.915, mAP@0.5:0.95=0.681. Supera la meta de tesis (>0.85). Comparar contra baseline del estado del arte: Pocevičė et al. (2024) con YOLOv5 sobre DeMSI, mAP@0.5=0.79. |
| Precisión y Recall | ✅ | yolov8m: P=0.889, R=0.866. Cumplen ≥80% de precisión (meta de tesis). |
| Comparativa yolov8m vs yolov8s | ✅ | Test de Wilcoxon (correcto para 2 modelos): F1 sin diferencia significativa (p=0.13); IoU con diferencia estadística pero despreciable en magnitud (0.8250 vs 0.8249, p=0.0047). **Usar este resultado para justificar la elección de yolov8m**: si el desempeño es equivalente, yolov8m ofrece mayor capacidad de generalización. Conectar con latencia: si en producción m ≈ s en ms/img, el argumento es completo. |

---

## 6. Métricas de identificación

> **Caveat que aplica a toda esta sección:** consultas sintéticas (variaciones 2D de diseños limpios). Etiquetar explícitamente en cada tabla y caption del paper. La validación con fotos reales sigue siendo la limitación principal.

| Punto | Estado | Detalle y acción |
|---|---|---|
| Top-1 / Top-5 / Top-10 | ✅ | Top-1=98.2%, Top-5=99.4%, Top-10=100.0% (galería producción, N=500 consultas sintéticas). Fuente: `reports/identification_topk_cmc_far_frr.md`. |
| mAP de retrieval | ✅ | mAP=0.988. Distinguir en el paper de mAP de detección (son métricas distintas). |
| FAR y FRR @ ρ | ✅ | ρ=0.55 (operación): FAR≈0.0%, FRR≈11.8%. EER≈11.1% en ρ≈0.32. Reportar en la Tabla II del paper (FAR/FRR con el umbral operativo elegido). |
| Umbral ρ y margen | ✅ | ρ=0.55, margen=0.05. Fuente: `src/embedding/match_config.json`. Calibrado sobre consultas sintéticas (Paso 0); pendiente recalibración con fotos reales. |
| Curva CMC y sweep de ρ (Fig. 3) | ✅ | `reports/cmc_curve.png` y `reports/far_frr_sweep.png` generados. Listos para incluir en el paper. |
| Ablación de la galería | ✅ | Separación banda/no-banda: DINOv2 solo +0.43 → +galería Blender +0.49 → +cabeza afinada +0.67. Documenta el aporte de cada componente. |
| Robustez a degradación | ✅ | Con cabeza afinada: Top-1=100% (limpio), 99% (40% degrad.), 90% (60%), 54% (80%). Ilustra el límite real del sistema bajo condiciones adversas. |
| Validación con fotos reales | ⚠️ | 6 fotos reales (cualitativo, N pequeño), incluyendo collage de Tren de Aragua con 5 coincidencias confirmadas. **Acción de mayor impacto:** obtener ~20-30 fotos reales etiquetadas para convertir este punto en una validación cuantitativa. Transforma la limitación principal del paper en un resultado real. Si no se alcanza antes del envío, declararlo como trabajo futuro con el protocolo exacto (`calibrate` + `compare_galleries --queries`). |

---

## 7. Consideraciones éticas y de seguridad

| Punto | Estado | Detalle y acción |
|---|---|---|
| Marco de herramienta de apoyo | ✅ | El sistema es explícitamente una herramienta de apoyo al análisis preliminar, no de identificación definitiva. Disclaimer permanente en la UI, `verdict` open-set en la API, lenguaje sin certeza absoluta en toda la interfaz. Documentar como decisión de diseño, no como limitación. |
| Cifrado y control de acceso | ⚙️ | HTTPS forzado en backend y frontend (Azure App Service + Static Web Apps). Cifrado en reposo AES-256 por defecto en Azure Blob Storage (verificar configuración al desplegar). Control de acceso por roles (Perito/Supervisor/Admin). Logs de acceso a imágenes (HU-13, pendiente). **Se documentará** en la subsección de seguridad del paper. |
| Audit log y trazabilidad forense | ⚙️ | El DBML ya incluye la tabla de auditoría (usuario, acción, timestamp, IP). **Se implementará** en TP2/TP3 (HU-14): logs append-only con retención mínima de 1 año. Mencionar en el paper como requisito de diseño cumplido en la arquitectura, con implementación en curso. |
| No reconocimiento facial | ⚙️ | El sistema analiza regiones de tatuaje, no rostros. Verificar que el dataset no contenga rostros identificables (ver §2). Declarar explícitamente en el paper. |

---

## 8. Pendientes críticos (resumen ejecutivo)

En orden de impacto sobre la credibilidad del paper:

1. **Validación con fotos reales (≥20-30)** — convierte la limitación más señalada en un resultado. Mayor ROI por unidad de esfuerzo.
2. **Criterios de curación de la galería** — sin esto, el ground truth de identificación no es defendible. Requiere fuente experta o criminológica citada.
3. ~~**Latencia en producción**~~ — **Hecho** (29/06/2026): stack completo desplegado en Azure (Container Apps + SQL serverless + Blob Storage), mediana 2.51s de procesamiento IA / 3.46s extremo a extremo, n=10. Ver §1.
4. **Política de anonimización** — verificar la muestra y redactar la subsección ética. Pedir aval al asesor si UPC lo exige.
5. **URL del repo** — decisión de visibilidad; si es público, limpiar historial y generar DOI en Zenodo.
