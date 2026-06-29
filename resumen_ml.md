📋 Resumen completo del proyecto ML — RTattooAI
Aquí tienes todo lo hecho, las métricas reales, lo que tenemos, y cómo conectarlo al backend/frontend. Todo está en la rama feat/yolo-detection-pipeline (pusheado).

Resumen ejecutivo
Sistema de 2 etapas, ambas funcionando y medidas:

Detección (YOLOv8) — encuentra y recorta el tatuaje. ✅ Entrenado, evaluado, exportado a ONNX.
Identificación (DINOv2 + cabeza afinada + galería) — compara el tatuaje contra una galería de bandas por similitud coseno, con rechazo open-set. ✅ Funcionando, con fine-tuning que mejora robustez.
Etapa 1 — Detección (YOLOv8)
Dataset: 7006 imágenes únicas / 8619 cajas (8 fuentes Roboflow, deduplicadas, 80/10/10).
Modelo definitivo yolov8m, exportado a ONNX (opset 12).
Métrica (test propio)	yolov8m	yolov8s
mAP@0.5	0.915	0.923
mAP@0.5:0.95	0.681	0.681
Precisión	0.889	0.893
Recall	0.866	0.873
Cumple la meta de la tesis (mAP@0.5 > 0.85, P ≥ 0.80).
Cross-dataset DeMSI (zero-shot): mAP@0.5 = 0.557 → brecha de dominio honesta (importante documentarla).
⚠️ Test pareado (Wilcoxon, 701 img de test, reports/stat_test_yolov8m_vs_yolov8s.md): yolov8m y yolov8s NO tienen diferencia significativa en F1 (p=0.13). La elección de yolov8m no está fuertemente respaldada por los datos — decirlo así de frente en el paper es más fuerte que ocultarlo.

Etapa 2 — Identificación (embeddings + galería)
DINOv2 (ViT-S/14) como extractor + cabeza afinada (contrastiva) + galería de 25 tatuajes de banda × ~17 vistas (diseño + Blender) — 25 confirmado como número DEFINITIVO.
Decisión = coseno vs galería, con umbral 0.55 + margen 0.05 (rechazo open-set).
Métrica	DINOv2 solo	+ galería Blender	+ cabeza afinada (actual)
Top-1 (consultas sintéticas)	94%	100%	~99%
Coseno del match verdadero	0.71	0.81	0.88
No-banda (falsos positivos)	0.28	0.32	0.21 ↓
Separación banda/no-banda	+0.43	+0.49	+0.67
Robustez a tatuajes degradados (con cabeza): top-1 = 100% (limpio) · 99% (40% degradación) · 90% (60%) · 54% (80% = casi irreconocible).

Métricas extendidas ya calculadas sobre la galería de producción (reports/identification_topk_cmc_far_frr.md): Top-5 = 99.4%, Top-10 = 100%, mAP de retrieval = 0.988, curva CMC completa (cmc_curve.png). FAR/FRR @ ρ=0.55: FAR≈0%, FRR≈11.8% (EER≈11.1% en ρ≈0.32) — el FRR lo causa sobre todo el margen, no el umbral (sweep en far_frr_sweep.png).

⚠️ Honesto: las métricas de arriba usan consultas sintéticas (no fotos reales) — son casi un techo, no una promesa de precisión en campo. Ya corrimos una primera validación REAL (N=6, reports/real_photo_validation.md) con el pipeline completo: 3/3 negativos correctos, 5/5 símbolos correctos en un collage real de prensa sobre Tren de Aragua (estrella/corona/AK47/reloj/locomotora), pero falló 1/2 en fotos de "payaso" (boceto artesanal muy distinto del diseño limpio, y un caso de margen bloqueando un top-1 correcto por ambigüedad con un diseño gemelo). Sigue faltando la validación definitiva con ~20-30 fotos reales etiquetadas. La detección sí está medida con datos reales desde el inicio.

Qué tenemos (artefactos)
Modelos (en models/, gitignored — locales):

yolov8m_best.onnx — detector para el backend .NET.
yolov8m_best.pt, yolov8s_best.pt — pesos.
embed_head.pt — cabeza afinada del embedding.
gallery_dinov2.npz — galería (vectores de los 25 + vistas).
Reportes (en reports/, sí versionados): métricas de detección, PR, matrices, ablación de galería, robustez a degradación, procedencia del dataset.

Scripts clave: detección (download/prepare/train/evaluate/export_onnx), identificación (embedder/identify/build_gallery/calibrate/finetune_head/...), demos Streamlit, pipeline Blender.

🔌 Cómo se usa (backend/frontend)
Detección → ya lista para .NET: carga yolov8m_best.onnx con ONNX Runtime. Entrada 1×3×640×640 (NCHW, RGB 0-1); salida [1,5,8400] (x,y,w,h,score); aplica NMS (conf≥0.25, iou≥0.45). Ejemplo en C# en el README.

Identificación → 2 opciones (te recomiendo la A por el deadline):

🟢 A. Microservicio Python (FastAPI) — rápido de montar, reusa el código que ya funciona.

Un servicio con 2 endpoints: POST /identify (imagen → detecciones + top-K bandas + veredicto) y POST /enroll (imagen + banda → la agrega a la galería con degradaciones).
Tu backend ASP.NET lo llama por HTTP; el frontend llama al .NET.
Reusa TattooIdentifier (ya probado). Es lo más realista para entregar en pocos días.
🔵 B. Todo en .NET (más "puro", más trabajo).

Exportar DINOv2 + cabeza a ONNX → embedding en .NET; coseno vs galería (vectores en BD) en C#. Requiere replicar el preprocesado (resize 518, normalización) exacto. Más riesgo bajo deadline.
La galería en producción: una tabla (banda, nombre_tatuaje, vector[128]). Matching = coseno vs todos → máx por banda → umbral 0.55 + margen 0.05 → top-K + veredicto. El estándar está en src/embedding/match_config.json.

Pendiente (honesto, para el informe)
Validar identificación con fotos reales a mayor escala (~20-30 fotos etiquetadas) — el N=6 ya corrido ayuda, pero no cierra la conclusión definitiva.
Reentrenar/fine-tune detección con DeMSI/imágenes peruanas (cierra la brecha de dominio) — para después.
Desplegar serving/ a Azure Container Apps con la cuenta real (script listo: serving/deploy/azure_deploy.sh, 1 vCPU/2GiB, scale-to-zero, capa gratis/estudiante) — falta correrlo para tener la latencia real de producción en vez de la medida localmente.
Decidir: URL del repo (hoy privado), política de anonimización (no existe paso de blur, las fuentes ya vienen recortadas), criterios expertos de curación de simbología.