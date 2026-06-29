#!/usr/bin/env bash
# Despliega serving/ a Azure Container Apps dentro de la capa gratis/estudiante.
#
# Pensado para correr en Git Bash / WSL / Azure Cloud Shell, desde la RAÍZ
# del repo (rtattooai-ml), porque el build usa serving/Dockerfile con
# contexto = raíz (necesita src/, serving/ y models/).
#
# Costos (referencia, revisa precios vigentes en azure.microsoft.com):
#   - Azure Container Apps (Consumption plan): SIEMPRE GRATIS hasta
#     180,000 vCPU-s + 360,000 GiB-s + 2,000,000 requests por mes, en
#     CUALQUIER tipo de cuenta (Free, Student, Pay-as-you-go). Con
#     min-replicas=0 (scale-to-zero) y tráfico de demo/tesis (no 24/7),
#     te quedas dentro de ese grant casi seguro.
#   - Azure Container Registry (SKU Basic): NO es parte del free-tier
#     "always free" — cuesta ~U$0.167/día (~U$5/mes) desde el día 1. Con
#     el crédito de Azure for Students (~U$100) esto no es un problema
#     para la duración de una tesis. Alternativa de costo CERO: usar un
#     repo público de Docker Hub en vez de ACR (ver nota al final).
#   - Log Analytics (logs del Container Apps env): capa gratis de 5GB de
#     ingesta/mes — de sobra para tráfico de demo.
#
# Uso:
#   az login
#   ./serving/deploy/azure_deploy.sh
#
# Variables override-ables, p. ej.:
#   ACR_NAME=miacrunico LOCATION=westus2 ./serving/deploy/azure_deploy.sh

set -euo pipefail

# ---- Configuración (override con variables de entorno) ----
RESOURCE_GROUP="${RESOURCE_GROUP:-rtattooai-rg}"
LOCATION="${LOCATION:-eastus2}"                     # ajusta si tu región no tiene Container Apps
ACR_NAME="${ACR_NAME:-rtattooaiacr$RANDOM}"         # debe ser único GLOBALMENTE en Azure
ENV_NAME="${ENV_NAME:-rtattooai-env}"
APP_NAME="${APP_NAME:-rtattooai-serving}"
IMAGE_NAME="${IMAGE_NAME:-rtattooai-serving}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CPU="${CPU:-1.0}"
MEMORY="${MEMORY:-2.0Gi}"
MIN_REPLICAS="${MIN_REPLICAS:-0}"                   # 0 = scale-to-zero (clave para quedarse gratis)
MAX_REPLICAS="${MAX_REPLICAS:-1}"
TARGET_PORT="${TARGET_PORT:-8000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "== Repo root: $REPO_ROOT =="
echo "== Resource group: $RESOURCE_GROUP | región: $LOCATION =="

# ---- 0) Extensión y providers ----
az extension add --name containerapp --upgrade -y >/dev/null
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

# ---- 1) Resource group (idempotente) ----
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none

# ---- 2) Azure Container Registry (Basic, ~U$5/mes) ----
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" \
        --sku Basic --admin-enabled true -o none
    echo "ACR creado: $ACR_NAME.azurecr.io"
else
    echo "ACR ya existe: $ACR_NAME"
fi

# ---- 3) Build remoto en ACR (no necesita Docker local) ----
echo "== Build de la imagen en ACR (puede tardar varios minutos: ~100MB de contexto) =="
az acr build --registry "$ACR_NAME" --resource-group "$RESOURCE_GROUP" \
    --image "$IMAGE_NAME:$IMAGE_TAG" --file serving/Dockerfile .

ACR_SERVER="$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)"
ACR_PASSWORD="$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)"
ACR_USERNAME="$(az acr credential show --name "$ACR_NAME" --query username -o tsv)"

# ---- 4) Entorno de Container Apps ----
if ! az containerapp env show --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    az containerapp env create --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" -o none
else
    echo "Entorno de Container Apps ya existe: $ENV_NAME"
fi

# ---- 5) Container App (crear o actualizar) ----
if ! az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    az containerapp create \
        --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
        --environment "$ENV_NAME" \
        --image "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
        --registry-server "$ACR_SERVER" \
        --registry-username "$ACR_USERNAME" --registry-password "$ACR_PASSWORD" \
        --target-port "$TARGET_PORT" --ingress external \
        --cpu "$CPU" --memory "$MEMORY" \
        --min-replicas "$MIN_REPLICAS" --max-replicas "$MAX_REPLICAS" \
        -o none
else
    echo "Container App ya existe, actualizando imagen…"
    az containerapp update \
        --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
        --image "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
        --cpu "$CPU" --memory "$MEMORY" \
        --min-replicas "$MIN_REPLICAS" --max-replicas "$MAX_REPLICAS" \
        -o none
fi

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv)"

echo ""
echo "== Listo =="
echo "URL: https://$FQDN"
echo "Prueba: curl https://$FQDN/health"
echo ""
echo "Para medir latencia real (reemplaza el benchmark local):"
echo "  for i in \$(seq 1 20); do curl -s -o /dev/null -w '%{time_total}\\n' \\"
echo "    -X POST https://$FQDN/identify -F image=@datasets/unified/images/test/<archivo>.jpg; done"
echo ""
echo "Para NO seguir generando ningún cargo, borra todo con:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
