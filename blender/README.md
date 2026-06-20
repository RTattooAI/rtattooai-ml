# Pipeline sintético de Blender (v7) — guía paso a paso

Genera, a partir de cada **diseño de tatuaje de banda**, muchas **vistas
realistas recortadas** (distinta piel, zona, ángulo, luz) para enriquecer la
**galería de identificación** (Etapa 2). No es para detección.

> El script corre **dentro de Blender** (usa `bpy`), no en el venv del repo.
> Suele hacerse en la máquina con la escena `.blend` (p. ej. la de Diego); luego
> se copian los recortes a la máquina del pipeline ML (la de Ricky).

---

## 0) Requisitos de la escena `.blend`

Antes de correr, la escena debe tener (los nombres deben coincidir con el config
al inicio de `generate_dataset_v7.py`; si difieren, renómbralos o edita el config):

- Maniquí: objeto **`Ch36`**.
- Cámara: objeto **`Cámara`**.
- Decals: **`TattooDecal_Pecho`**, **`TattooDecal_Espalda`**, **`TattooDecal_Bicep1`**,
  **`TattooDecal_Cintura1`**. Cada uno con material que tenga un nodo
  **Image Texture** + un nodo **Mapping**.
- Materiales de piel: **`Skin_Light`**, **`Skin_Medium`**, **`Skin_Dark`**.
- Diseños PNG en `BASE_PATH/Tatuajes_Horizontal`, `.../Tatuajes_Vertical`,
  `.../Tatuajes_Compacto`.

**Edita al inicio del script:**
```python
BASE_PATH   = r"...ruta a IMAGENES TEXTURA DATASET TATUAJES"
OUTPUT_PATH = r"...ruta donde guardar OUTPUT"
```

---

## 1) Primera corrida en MODO TEST (rápido)

El script ya viene en modo test (1 tatuaje por categoría, piel `medium`, ángulo
`frontal`):
```python
MAX_TATTOOS_PER_CATEGORY = 1
ONLY_SKINS = ["medium"]
ONLY_ANGLES = ["frontal"]
```

Pasos:
1. Abre la escena `.blend`.
2. Ve al workspace **Scripting**.
3. **Window → Toggle System Console** (para ver los `print`).
4. Abre `generate_dataset_v7.py` (o pégalo en un Text nuevo).
5. Ejecuta con **Run Script** (o `Alt+P`).

---

## 2) Verifica el resultado del test (3 cosas)

1. **Recortes, no cuerpo entero.** Abre `OUTPUT/crops/<tattoo_id>/...png`: debe
   verse el **tatuaje encuadrado**, no el maniquí completo.
   - Si sale el cuerpo entero → el render por región no se aplicó (revisa versión
     de Blender; `use_crop_to_border` existe en 2.8+).
   - Si el recorte tiene **mucha piel** alrededor → baja `CROP_MARGIN`.
     Si **corta** el tatuaje → súbelo. (Si el plano del decal es mayor que el
     tatuaje por el zoom del Mapping, el recorte puede quedar holgado.)
2. **Visibilidad correcta.** Con cámara **frontal**: `pecho` debe renderizar y
   `espalda` debe **saltarse** (mira al lado opuesto). Mira el conteo
   "saltados" al final.
   - Si pasa al revés (renderiza espalda vacía o salta pecho) → el signo de
     `FACING_THR` está invertido para tus decals: cambia `-0.10` por `+0.10`,
     o pon `VISIBILITY_CHECK = False` para desactivarlo.
3. **`OUTPUT/manifest.csv`** tiene filas (crop, identidad, zona, piel, ángulo…).

Ajusta el config y repite el test hasta que se vea bien.

---

## 3) Corrida de PRODUCCIÓN

Cambia a:
```python
MAX_TATTOOS_PER_CATEGORY = None
ONLY_SKINS = None
ONLY_ANGLES = None
```
Combinatoria aprox.: 25 tatuajes × 6 zonas × 3 pieles × 5 ángulos ≈ **2 250
renders** (menos los no visibles). Recomendaciones:
- Usa **EEVEE** (mucho más rápido que Cycles) si quieres volumen.
- Para no bloquear Blender, corre **headless** desde terminal:
  ```bash
  blender escena.blend --background --python blender/generate_dataset_v7.py
  ```
- El `RANDOM_SEED` fija la randomización (reproducible).

---

## 4) De vuelta en el repo (máquina del pipeline ML)

1. **Copia** `OUTPUT/crops` a la máquina del repo, p. ej. a
   `datasets/blender_crops/` (gitignored).
2. **Galería multi-vista** (diseño limpio + vistas Blender):
   ```bash
   python -m src.embedding.build_gallery --sources datasets/gang_tattoos datasets/blender_crops
   ```
3. **Ablación para la tesis** (cuánto aporta el sintético):
   ```bash
   python -m src.embedding.compare_galleries \
       --gallery-a datasets/gang_tattoos \
       --gallery-b datasets/gang_tattoos datasets/blender_crops \
       --synthetic-queries 6          # o --queries <fotos_reales> cuando las tengas
   ```
   Reporte en `reports/gallery_ablation.md` + `.png`.
4. **Probar** el flujo completo: `streamlit run app/streamlit_full.py`.
5. **Calibrar el umbral** con fotos reales (cuando las tengas):
   ```bash
   python -m src.embedding.calibrate --positives <reales> --negatives <no_banda> --write
   ```

---

## Problemas comunes

| Síntoma | Causa / solución |
|---|---|
| "Falta maniquí/cámara/decal" | Los nombres no coinciden con el config → renombra o edita el config |
| Recortes con cuerpo entero | Render por región no aplicado → revisa versión de Blender |
| Se salta todo / nada | Signo de `FACING_THR` invertido → cámbialo o `VISIBILITY_CHECK=False` |
| Recortes negros/vacíos | Tatuaje no visible o material sin Image Texture cargado |
| Recorte muy holgado/apretado | Ajusta `CROP_MARGIN` |
