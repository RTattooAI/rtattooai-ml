# RTattooAI — Resumen de métricas finales y pipelines ML

> Documento consolidado de los resultados más relevantes del repositorio `rtattooai-ml`.
> 
> **Aviso importante:** RTattooAI es una herramienta de apoyo al análisis preliminar. Ningún resultado debe tratarse como identificación definitiva; siempre requiere validación humana.

## 1. Visión general

El proyecto tiene dos etapas de IA:

1. **Detección de tatuajes** con YOLOv8.
2. **Identificación por similitud** con embeddings DINOv2 + galería de referencia.

Además, existe un pipeline sintético en **Blender** para generar vistas realistas de tatuajes y enriquecer la galería de identificación.

---

## 2. Métricas finales disponibles

### 2.1 Detección YOLOv8 sobre el dataset propio

Métricas reportadas en `reports/metrics.md`.

| Modelo | Split | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---:|---:|---:|---:|
| yolov8m | val | 0.9237 | 0.7184 | 0.9311 | 0.8754 |
| yolov8m | test | 0.9149 | 0.6810 | 0.8888 | 0.8657 |
| yolov8s | val | 0.9164 | 0.7008 | 0.8915 | 0.8803 |
| yolov8s | test | 0.9227 | 0.6809 | 0.8928 | 0.8726 |

**Modelo definitivo:** `yolov8m`

- `mAP@0.5 = 0.915`
- `Precisión = 0.889`
- Cumple la meta del proyecto: `mAP@0.5 > 0.85` y `precisión >= 0.80`.
- Supera el baseline de referencia de Pocevičė et al. 2024 (`0.79`).

### 2.2 Detección cross-dataset sobre DeMSI

Métricas reportadas en `reports/metrics_demsi.md`.

| Modelo | Split | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---:|---:|---:|---:|
| yolov8m | val | 0.5565 | 0.3896 | 0.8226 | 0.5174 |
| yolov8s | val | 0.5619 | 0.3944 | 0.7917 | 0.5371 |

**Lectura:**

- La precisión se mantiene razonable.
- El recall cae bastante, lo que confirma una brecha de dominio clara entre el dataset propio y DeMSI.
- Esta evaluación es útil para documentar la generalización, pero no es comparable de forma directa con el baseline de referencia porque el baseline fue entrenado sobre DeMSI.

### 2.3 Robustez del embedding ante degradación

Métricas de `reports/robustness_degradation.md`.

| Nivel de degradación | Coseno vs diseño limpio | Top-1 |
|---|---:|---:|
| 0.0 | 1.000 | 100.0% |
| 0.2 | 0.968 | 100.0% |
| 0.4 | 0.923 | 99.0% |
| 0.6 | 0.852 | 90.0% |
| 0.8 | 0.677 | 54.0% |

**Conclusión:** el sistema mantiene `top-1 >= 80%` hasta un nivel de degradación de `0.6`.

### 2.4 Ablación de galería multi-vista

Métricas de `reports/gallery_ablation.md`.

| Configuración | Top-1 | Coseno genuino medio | Coseno no-banda medio | Brecha |
|---|---:|---:|---:|---:|
| Galería A, solo diseño | 95.3% | 0.876 | 0.211 | +0.666 |
| Galería B, multi-vista | 99.3% | 0.909 | 0.252 | +0.657 |

**Conclusión:** añadir vistas sintéticas y realistas a la galería mejora la recuperación `top-1` y hace más robusto el matching.

### 2.5 Baseline de identificación por embeddings

Métricas de `reports/embedding_step0.md`.

- Galería: 25 tatuajes.
- Consultas sintéticas por tatuaje: 8.
- Embedding: DINOv2 ViT-S/14.
- Resultado orientativo: `top-1 ≈ 91%`.
- Separación genuino vs no-banda suficiente para fijar un umbral inicial de rechazo open-set.

**Nota:** estas consultas son sintéticas, así que sirven para validar la mecánica, no para cerrar conclusiones definitivas.

### 2.6 Inferencia externa sobre DeMSI sin anotaciones

Reporte cualitativo en `reports/external_demsi/summary.md`.

- Imágenes procesadas: `1451`
- Imágenes con al menos una detección: `1388 (95.7%)`
- Cajas detectadas en total: `2050`
- Confianza máxima por imagen: media `0.852`, mediana `0.897`

**Nota:** al no haber anotaciones de verdad, esto no es una métrica de precisión/recall real.

### 2.7 Top-k, CMC y mAP de retrieval (galería de producción)

Métricas de `reports/identification_topk_cmc_far_frr.md` (script
`src/embedding/identification_metrics.py`). Usa la galería ACTUAL de
producción (25 identidades × 17 vistas, embedding con cabeza afinada) y
500 consultas sintéticas (20 por diseño).

| Métrica | Valor |
|---|---:|
| Top-1 | 98.2% |
| Top-5 | 99.4% |
| Top-10 | 100.0% |
| mAP de retrieval | 0.988 |

Curva CMC completa en `reports/cmc_curve.png`.

**Nota:** consultas sintéticas (variaciones 2D de los diseños limpios) —
son casi un techo porque parten de las mismas identidades ya vistas en la
galería. Funcionan como sanity-check del espacio de embeddings, no como
estimación de precisión en campo (ver §2.10).

### 2.8 FAR / FRR en función del umbral ρ

Mismo script y reporte que §2.7; sweep completo en `reports/far_frr_sweep.png`.

| ρ (umbral) | FAR | FRR |
|---:|---:|---:|
| 0.45 | 2.7% | 11.8% |
| 0.50 | 0.0% | 11.8% |
| **0.55 (operación actual)** | **0.0%** | **11.8%** |
| 0.60 | 0.0% | 11.8% |
| 0.70 | 0.0% | 14.2% |

- **EER ≈ 11.1%** en ρ ≈ 0.32.
- El FRR se mantiene casi plano entre ρ=0.30 y ρ=0.60 → el rechazo lo está
  causando el **margen** (0.05), no el umbral: cuando hay dos diseños muy
  parecidos en la galería, el margen bloquea el match aunque el top-1 esté
  bien rankeado y sobre el umbral (confirmado en un caso real, ver §2.10).

### 2.9 Test estadístico pareado: yolov8m vs yolov8s

Reporte en `reports/stat_test_yolov8m_vs_yolov8s.md` (script
`src/detection/compare_models_stat.py`). Wilcoxon signed-rank sobre las
701 imágenes de test, mismo set para ambos modelos (Friedman no aplica con
solo 2 modelos).

| Métrica por imagen | yolov8m | yolov8s | Wilcoxon |
|---|---:|---:|---|
| F1 @ IoU≥0.5 | 0.9042 ± 0.2525 | 0.9116 ± 0.2416 | p = 0.1315 → **sin diferencia significativa** |
| IoU medio (TPs) | 0.8250 ± 0.2270 | 0.8249 ± 0.2176 | p = 0.0047 → significativa, pero la diferencia (0.0001) es **irrelevante en la práctica** |

**Lectura honesta:** `yolov8m` y `yolov8s` están estadísticamente empatados
en este test set. La elección de `yolov8m` como modelo definitivo no está
fuertemente respaldada por los datos — `yolov8s` es una alternativa válida
si la latencia en CPU pesa más que el ~0.5pt de mAP.

### 2.10 Validación preliminar con fotos reales (N=6)

Reporte completo en `reports/real_photo_validation.md`. Pipeline de
producción completo (`identify.py`) sobre 6 fotos reales (3 positivas, 3
negativas), incluyendo un collage periodístico real de tatuajes de Tren de
Aragua.

- **3/3 negativos correctamente rechazados** (sin falsos positivos).
- **5/5 símbolos identificables del collage correctamente identificados**
  (estrella, corona, AK-47, reloj, locomotora) sobre fotos reales de prensa,
  no sintéticas.
- **1/2 fotos de "payaso" fallaron**: una por estilo de tatuaje muy distinto
  al diseño limpio de catálogo (boceto artesanal); otra por ambigüedad entre
  dos diseños casi gemelos de la galería, bloqueada por el margen (top-1
  correcto y sobre el umbral, pero rechazada por margen — comportamiento
  conservador, no un error de ranking).

**Nota:** N pequeño, no sustituye la validación definitiva con ~20-30 fotos
reales etiquetadas, pero es la primera evidencia REAL (no sintética) del
sistema y matiza fuertemente los números de §2.7.

### 2.11 Latencia del pipeline (local, proxy de producción)

Reporte en `reports/serving_latency_local.md` (script
`serving/scripts/benchmark_latency.py`). Mide `processing_ms` end-to-end
(detección + identificación) sobre CPU local — todavía no hay Container
App real desplegado (ver `serving/deploy/`).

| Escenario | Mediana | p95 | Cold start |
|---|---:|---:|---:|
| Multi-hilo (CPU local completa) | 944 ms | 2843 ms | hasta ~9.3s |
| 1 hilo (proxy de 1 vCPU Azure) | 1205 ms | 2455 ms | ~0.6s (caché en proceso) |

**Nota:** son cifras locales, no del entorno final. `serving/deploy/azure_deploy.sh`
despliega a un Container App de 1 vCPU/2 GiB con scale-to-zero (capa gratis)
e imprime el comando para volver a medir contra la URL real.

---

## 3. Pipeline de detección YOLOv8

### 3.1 Objetivo

Detectar tatuajes en imágenes con una sola clase: `tatuaje`.

### 3.2 Flujo del pipeline

1. Reunir datasets de múltiples fuentes.
2. Normalizar anotaciones a una sola clase.
3. Convertir polígonos o máscaras a bounding boxes cuando corresponda.
4. Eliminar duplicados entre fuentes.
5. Separar en `train / val / test`.
6. Entrenar YOLOv8.
7. Evaluar en `val` y `test`.
8. Exportar el modelo final a ONNX.

### 3.3 Scripts clave

- `src/detection/download_data.py`: descarga las fuentes.
- `src/detection/prepare_dataset.py`: unifica, normaliza, deduplica y divide el dataset.
- `src/detection/train.py`: entrena `yolov8m` y `yolov8s`.
- `src/detection/evaluate.py`: calcula métricas y genera reportes.
- `src/detection/export_onnx.py`: exporta el modelo definitivo a ONNX.

### 3.4 Dataset final de entrenamiento

- `7006` imágenes únicas.
- `8619` cajas.
- Split aproximado `80/10/10`.
- Dedupe con hash perceptual para evitar fuga entre fuentes.

### 3.5 Modelo final

- **Modelo definitivo:** `yolov8m`
- **Baseline comparativo:** `yolov8s`
- Parámetros principales:
  - `imgsz = 640`
  - `epochs = 150`
  - `patience = 30`
  - `batch = auto`
  - semilla fija `42`

### 3.6 Resultado operativo

El modelo definitivo `yolov8m` quedó exportado para consumo del backend como `yolov8m_best.onnx`.

---

## 4. Pipeline de identificación por embeddings

### 4.1 Objetivo

Tomar el recorte del tatuaje detectado y compararlo contra una galería de referencia para proponer una coincidencia por similitud coseno.

### 4.2 Flujo del pipeline

1. El detector YOLO localiza el tatuaje.
2. Se recorta la región detectada.
3. Se extrae un embedding con DINOv2.
4. Se compara contra la galería de referencia.
5. Se ordenan los candidatos por similitud.
6. Se aplica un umbral y un margen para decidir `match` o `no_match`.

### 4.3 Componentes principales

- `src/embedding/embedder.py`: extractor DINOv2 y cabeza opcional afinada.
- `src/embedding/step0_baseline.py`: baseline de identificación sin entrenar.
- `src/embedding/build_gallery.py`: construye la galería multi-vista.
- `src/embedding/identify.py`: identifica un tatuaje en una imagen.
- `src/embedding/calibrate.py`: calibra umbral y margen con datos reales.
- `src/embedding/finetune_head.py`: afina una cabeza contrastiva sobre DINOv2.
- `src/embedding/robustness_test.py`: mide robustez a degradación.

### 4.4 Qué hace cada pieza

**DINOv2 / embedder**

- Usa `DINOv2 ViT-S/14` como extractor base.
- El embedding final se normaliza para que la similitud sea coseno.
- Puede cargar una cabeza MLP afinada para mejorar invarianza.

**Galería**

- Cada identidad puede tener varias vistas.
- La similitud de una identidad se toma como el coseno máximo entre sus vistas.
- La galería se guarda en `models/gallery_dinov2.npz`.

**Calibración**

- Busca el mejor par `umbral + margen` sobre fotos reales.
- Escribe `match_config.json` con el estándar de operación.

**Fine-tuning de cabeza**

- Congela DINOv2.
- Entrena una MLP pequeña con pérdida contrastiva supervisada.
- El resultado se guarda como `models/embed_head.pt`.

### 4.5 Resultado práctico

El pipeline de embeddings está pensado para apoyar el juicio humano con top-K de candidatos, no para cerrar una identificación automática definitiva.

---

## 5. Pipeline de Blender

### 5.1 Objetivo

Generar vistas realistas de tatuajes de banda para enriquecer la galería de identificación.

### 5.2 Qué produce

- Recortes del tatuaje sobre piel 3D.
- Variación de zona corporal, piel, ángulo y condiciones visuales.
- `OUTPUT/crops/<tattoo_id>/*.png`
- `OUTPUT/manifest.csv`

### 5.3 Requisitos de escena

La escena `.blend` debe incluir, con nombres coherentes:

- Maniquí: `Ch36`
- Cámara: `Cámara`
- Decals: `TattooDecal_Pecho`, `TattooDecal_Espalda`, `TattooDecal_Bicep1`, `TattooDecal_Cintura1`
- Materiales de piel: `Skin_Light`, `Skin_Medium`, `Skin_Dark`

### 5.4 Flujo del pipeline

1. Cargar la escena Blender.
2. Asignar diseños PNG a los decals.
3. Renderizar distintas combinaciones de piel, zona y ángulo.
4. Recortar automáticamente el área del tatuaje.
5. Guardar los crops y el manifiesto.
6. Copiar los crops al workspace ML para construir la galería.
7. Recalcular la galería multi-vista.
8. Recalibrar el umbral con fotos reales cuando estén disponibles.

### 5.5 Scripts clave

- `blender/generate_dataset_v7.py`: pipeline principal de Blender.
- `blender/generate_synthetic_procedural.py`: generación sintética procedural.

### 5.6 Uso en el sistema

Los crops de Blender se usan como vistas adicionales para la galería de embeddings. Su valor no está en la detección, sino en mejorar la identificación por similitud.

---

## 6. Pipeline de inferencia de extremo a extremo

### 6.1 Servicio interno

El directorio `serving/` encapsula el flujo completo:

1. Detectar tatuajes con YOLOv8 ONNX.
2. Recortar cada detección.
3. Obtener embeddings con DINOv2.
4. Buscar coincidencias en la galería.
5. Devolver veredicto y top-K.

### 6.2 Artefactos usados

- `models/yolov8m_best.onnx`
- `models/embed_head.pt`
- `models/gallery_dinov2.npz`
- `src/embedding/match_config.json`

### 6.3 Regla de veredicto

- `match` si la mejor similitud supera el umbral y además supera al segundo mejor por el margen definido.
- `no_match` si no cumple esas dos condiciones.

---

## 7. Lectura rápida de la solución

- **Detección:** `yolov8m` cumple la meta principal, pero el test pareado
  (§2.9) muestra que está estadísticamente empatado con `yolov8s` — la
  elección es defendible por mAP en val, no por significancia estadística.
- **Identificación:** DINOv2 + galería multi-vista es el núcleo del segundo
  paso. Los números sintéticos (Top-1 98.2%, mAP 0.988) son casi un techo;
  la evidencia real (§2.10) es más matizada: funciona muy bien con símbolos
  icónicos de un solo trazo, es conservadora ante diseños casi gemelos en la
  galería, y le cuesta más con interpretaciones artesanales muy distintas
  del diseño de catálogo.
- **Blender:** aporta diversidad visual y hace la galería más robusta.
- **Servidor:** une detección e identificación; listo para desplegar a
  Azure Container Apps en capa gratis/estudiante (`serving/deploy/`).

---

## 8. Fuentes de referencia dentro del repo

- `reports/metrics.md`
- `reports/metrics_demsi.md`
- `reports/robustness_degradation.md`
- `reports/gallery_ablation.md`
- `reports/embedding_step0.md`
- `reports/external_demsi/summary.md`
- `reports/identification_topk_cmc_far_frr.md` (+ `.json`, `cmc_curve.png`, `far_frr_sweep.png`)
- `reports/stat_test_yolov8m_vs_yolov8s.md`
- `reports/real_photo_validation.md`
- `reports/serving_latency_local.md`
- `reports/paper_requirements_checklist.md`
- `README.md`
- `blender/README.md`
- `src/embedding/README.md`
- `serving/README.md`
- `serving/deploy/README.md`

---

## 9. Conclusión

Si lo que se busca es el estado más fiel y utilizable del proyecto ML, la lectura correcta es esta:

- La **detección YOLOv8** ya está cerrada con métricas sólidas y exportación
  a ONNX, aunque `yolov8m` vs `yolov8s` no tiene diferencia estadísticamente
  significativa en F1 (§2.9) — vale la pena decirlo así en el paper.
- La **identificación por embeddings** está operativa y mejora con galería
  multi-vista y fine-tuning de cabeza; los Top-k/CMC/mAP/FAR-FRR (§2.7-2.8)
  ya están calculados sobre la galería de producción, pero siguen siendo
  sintéticos — la validación real preliminar (§2.10, N=6) es la evidencia
  que más pesa y matiza los números sintéticos.
- **Blender** es un componente auxiliar importante para aumentar variedad y robustez.
- El **despliegue** ya tiene script listo para Azure Container Apps en capa
  gratis/estudiante (1 vCPU/2 GiB, scale-to-zero); falta correrlo con la
  cuenta real para reemplazar la latencia local (§2.11) por la de producción.
- La gran brecha pendiente sigue siendo la **validación con fotos reales a
  mayor escala** (~20-30 fotos etiquetadas) y la **generalización cross-domain**.
