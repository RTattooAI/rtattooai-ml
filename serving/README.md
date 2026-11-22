# serving/ — Servicio de inferencia (FastAPI)

Microservicio interno (consumido por el backend ASP.NET Core, no expuesto a
internet) que envuelve el pipeline de detección + identificación ya
entrenado: **YOLOv8m (ONNX) + DINOv2 ViT-S/14 + cabeza afinada + galería**.
No reentrena nada; ver `API_CONTRACT.md` sección A para el contrato exacto.

## Modo ensamble (por defecto desde el 23/10/2026)

Desde esta versión el servicio identifica con un **ensamble de dos codificadores
con verificación por consenso y semáforo conformal** (`MODO_IDENTIFICACION=ensamble`,
o `"modo"` en `match_config.json`). El modo anterior sigue disponible con
`MODO_IDENTIFICACION=clasico` para volver atrás sin reconstruir la imagen.

| Pieza | Archivo | Qué hace |
|---|---|---|
| Generalista | `models/ensamble/generalista.pt` | Cabeza contrastiva sobre DINOv2 ViT-S/14 |
| Especialista | `models/ensamble/especialista.pt` | CNN (ResNet) afinada con tatuajes |
| Galería base | `models/ensamble/galeria_inicial.npz` | Catálogo de símbolos con los dos vectores y vistas giradas |
| Calibración | `models/ensamble/calibracion_desconocidos.npz` | Tatuajes que no están en la galería |
| Lógica | `serving/ensamble.py` | Normalización, línea conformal y veredicto (sin GPU, con pruebas) |

**Cómo decide** (detalle en el docstring de `serving/ensamble.py`):

1. Cada codificador puntúa cada identidad (coseno máximo entre sus vistas).
2. Cada modelo se normaliza con sus puntajes contra los desconocidos de calibración
   y se promedian con el mismo peso: **no hay pesos elegidos a mano**.
3. La línea τ se calcula para que, como máximo, una fracción **α** de los
   desconocidos la pase (`"alfa"` en `match_config.json`, 0.05 por defecto). Se
   recalcula sola al enrolar o quitar, porque depende del tamaño de la galería.
4. Veredicto: ningún candidato sobre τ → `no_match`; uno solo **y los dos modelos
   eligen ese mismo** → `match`; en otro caso → `review`.

**Escala que se muestra.** `similarity` es un puntaje calibrado: **0.70 = la línea
τ** y 0.55 = medio punto normalizado por debajo. Así la pantalla (que pinta verde
desde 0.70 y usa los cortes 70/55) dice lo mismo que el modelo, sin cambios en el
backend ni en el frontend. El coseno real viaja en `cosine`, el puntaje con el que
se decide en `score` y si pasa la línea en `candidate`; la detección trae
`consensus`. Son campos opcionales: el backend actual los ignora.

**Regenerar los artefactos** (al cambiar de pesos):

```bash
python -m src.embedding.construir_ensamble --generalista <cabeza.pt> --especialista <cnn.pt>
```

La galería del ensamble que crece al enrolar vive en `models/galeria_ensamble.npz`
(en Docker, dentro del volumen) o en el blob `galeria_ensamble.npz`; la primera vez
se siembra con la galería base. Tras regenerar, borre esa galería o use «Reparar
galería» en la pantalla de Modelos para re-enrolar lo que se añadió desde la app.

**Elegir y medir** (ver `reports/experimentos/`):

```bash
python -m src.embedding.validar_ensamble --generalistas ... --especialistas ...   # elegir (validación)
python -m src.embedding.evaluar_ensamble                                           # medir (prueba final)
python -m unittest serving.tests.test_ensamble serving.tests.test_gallery_store
```

## Artefactos que usa en modo clásico

- `models/yolov8m_best.onnx` — detector (CPU, vía `ultralytics.YOLO` + onnxruntime).
- `models/embed_head.pt` — cabeza afinada (384 → 256 → 128) sobre DINOv2.
- `models/gallery_dinov2.npz` — galería persistente (embeddings + `catalog_ids` + `bands`).
- `src/embedding/match_config.json` — `det_conf`, `det_iou`, `match_threshold`, `match_margin`.

Si `models/gallery_dinov2.npz` todavía tiene el esquema antiguo (`embeddings` +
`names`, sin `catalog_ids`/`bands`), correr primero:

```bash
python -m serving.scripts.migrate_gallery
```

## Correr en local

```bash
pip install -r serving/requirements.txt
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/identify \
  -F "image=@datasets/unified/images/test/by_tatoo__006307.jpg" \
  -F "top_k=5"

curl -X POST http://localhost:8000/enroll \
  -F "image=@datasets/gang_tattoos/Corona-removebg-preview.png" \
  -F "band=Los Pulpos"
```

## Variables de entorno

| Variable | Default | Uso |
|---|---|---|
| `PORT` | `8000` | Puerto del servidor (Azure Container Apps lo inyecta). |
| `DETECTOR_WEIGHTS_PATH` | `models/yolov8m_best.onnx` | Pesos del detector. |
| `HEAD_PATH` | `models/embed_head.pt` | Cabeza afinada del embedder. |
| `GALLERY_LOCAL_PATH` | `models/gallery_dinov2.npz` | Caché local de la galería. |
| `AZURE_STORAGE_CONNECTION_STRING` | _(vacío)_ | Si se define, la galería persiste en Blob (crece sin redesplegar). Sin ella, sólo el archivo local. |
| `GALLERY_CONTAINER` | `models` | Contenedor de Blob. |
| `GALLERY_BLOB_NAME` | `gallery_dinov2.npz` | Nombre del blob. |
| `DET_CONF`, `DET_IOU`, `MATCH_THRESHOLD`, `MATCH_MARGIN` | valores de `match_config.json` | Overrides opcionales del estándar de operación. |

## Docker (Azure Container Apps)

```bash
docker build -f serving/Dockerfile -t rtattooai-serving .
docker run -p 8000:8000 --env-file .env rtattooai-serving
```

## Notas

- La galería crece en runtime: cada `/enroll` agrega vistas degradadas (mismo
  generador de `src/embedding/degrade.py`) a la identidad nueva, recalcula y
  guarda — local y/o Blob, según configuración.
- `verdict` por detección sigue la regla de `src/embedding/identify.py`:
  `match` si top-1 ≥ `threshold` **y** (top-1 − top-2) ≥ `margin`; si no,
  `no_match`, devolviendo igual las coincidencias más cercanas (no
  concluyentes).
