# serving/ — Servicio de inferencia (FastAPI)

Microservicio interno (consumido por el backend ASP.NET Core, no expuesto a
internet) que envuelve el pipeline de detección + identificación ya
entrenado: **YOLOv8m (ONNX) + DINOv2 ViT-S/14 + cabeza afinada + galería**.
No reentrena nada; ver `API_CONTRACT.md` sección A para el contrato exacto.

## Artefactos que usa (ya existentes en `models/`)

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
