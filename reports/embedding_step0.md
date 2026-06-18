# Paso 0 — Baseline de identificación por embeddings (DINOv2, sin entrenar)

- Galería: **25** tatuajes de banda | variaciones por tatuaje: **8** (total 200 consultas).
- Embedding: DINOv2 ViT-S/14 (384-d), similitud coseno.

## 1-2) Recuperación (¿reconoce el tatuaje pese a las condiciones?)
- **Top-1: 91.5%** | **Top-5: 97.5%**
- Coseno consulta↔correcto: media **0.702** (min 0.075)
- Coseno consulta↔mejor-incorrecto: media **0.483**
- Brecha media correcto−incorrecto: **+0.219** (positiva = separa identidades)

## 3) Rechazo open-set (¿distingue banda de no-banda?)
- Distractores no-banda: 60 (recortados del test).
- Mejor-coseno consulta genuina: media **0.708**
- Mejor-coseno distractor no-banda: media **0.275**
- Umbral sugerido ≈ **0.492** → aceptaría 92% de genuinas y 2% de no-banda (falsos).
