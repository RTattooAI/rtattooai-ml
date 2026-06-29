# Despliegue a Azure Container Apps (capa gratis / Azure for Students)

Despliega `serving/` (FastAPI: detección YOLOv8m ONNX + identificación
DINOv2 + galería) a Azure Container Apps con un dimensionamiento pensado
para quedarse dentro de la capa gratis o del crédito de Azure for Students.

## Por qué este dimensionamiento

- **1 vCPU / 2 GiB por réplica**: con menos memoria el proceso (FastAPI +
  PyTorch + onnxruntime + DINOv2 + YOLO) queda muy ajustado; con menos vCPU
  la latencia medida localmente (`reports/serving_latency_local.md`,
  proxy de 1 vCPU: ~1.2s/imagen de media) se degrada más.
- **`min-replicas=0` (scale-to-zero)**: Container Apps (plan Consumption)
  da gratis, en CUALQUIER tipo de cuenta, 180,000 vCPU-segundos + 360,000
  GiB-segundos + 2,000,000 requests **por mes**. Si dejaras 1 vCPU corriendo
  24/7 todo el mes, usarías ~2,592,000 vCPU-segundos — muy por encima del
  grant gratis (~$58/mes adicionales). Con `min-replicas=0`, solo se cobra
  mientras hay tráfico real (demos, pruebas, defensa de tesis), así que es
  muy probable que te quedes dentro de la capa gratis.
- **Trade-off conocido**: con scale-to-zero, el primer request después de
  estar inactivo paga el "cold start" completo (cargar YOLOv8m ONNX +
  DINOv2 + galería) — varios segundos. Para una herramienta real de campo
  esto no sería aceptable; para el alcance de la tesis es una limitación
  documentada, no un blocker.

## Qué NO es gratis

- **Azure Container Registry (Basic)**: ~U$5/mes desde el día 1 (no tiene
  capa "always free"). Con el crédito de Azure for Students (~U$100) no es
  un problema durante la tesis. Si quieres costo CERO incluso en esto,
  alternativa: construir la imagen con `docker build` localmente y subirla
  a un repo PÚBLICO de Docker Hub en vez de ACR, y usar
  `--registry-server docker.io` en `az containerapp create`. Contras: el
  registro queda público (cualquiera puede descargar la imagen, que incluye
  los pesos de los modelos — no hay credenciales ni datos de bandas reales
  dentro, así que no es un riesgo de seguridad, pero sí de propiedad/difusión
  del modelo).
- **Log Analytics**: capa gratis de 5GB de ingesta/mes — de sobra para
  tráfico de demo, pero si haces pruebas de carga intensas podría agotarse.

## Prerrequisitos

1. Cuenta de Azure for Students activada (o Azure Free) y `az login` hecho.
2. `az` CLI instalado (`az --version`).
3. Los artefactos en `models/` deben existir localmente antes de correr el
   script (`yolov8m_best.onnx`, `embed_head.pt`, `gallery_dinov2.npz`) —
   el build los empaqueta dentro de la imagen (ver `serving/Dockerfile`).

## Uso

```bash
az login
./serving/deploy/azure_deploy.sh
```

Variables opcionales (export antes de correr, o inline):
`RESOURCE_GROUP`, `LOCATION`, `ACR_NAME` (debe ser único global), `ENV_NAME`,
`APP_NAME`, `CPU`, `MEMORY`, `MIN_REPLICAS`, `MAX_REPLICAS`.

Al final imprime la URL pública y el comando para medir latencia real (para
reemplazar el proxy local de `reports/serving_latency_local.md` con el
número real de producción que pide el paper).

## Apagar todo (evitar cualquier cargo)

```bash
az group delete --name rtattooai-rg --yes --no-wait
```

## Una vez integrado con el backend ASP.NET

Por ahora el ingress queda `external` para poder probarlo con `curl`. Cuando
el backend ya lo consuma y no necesites pegarle desde afuera, cambia a:

```bash
az containerapp ingress update --name rtattooai-serving \
    --resource-group rtattooai-rg --type internal
```
