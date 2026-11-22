#!/usr/bin/env bash
# Tras la cola: validar parejas -> elegir -> construir artefactos -> evaluación final.
cd "$(dirname "$0")/../../.."
PY=venv/Scripts/python.exe
L=reports/experimentos/candidatos
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf8
until grep -q "COLA TERMINADA" $L/cola.log; do sleep 60; done
echo "=== $(date +%H:%M) validación"
HEADS=$(ls models/candidatos/head_*.pt); CNNS=$(ls models/candidatos/cnn_*.pt)
$PY -m src.embedding.validar_ensamble --generalistas $HEADS --especialistas $CNNS > $L/log_validacion.txt 2>&1 || { echo FALLO validacion; exit 1; }
MEJOR=$($PY -c "import json; print(json.load(open('$L/validacion.json',encoding='utf8'))['mejor'])")
G="models/candidatos/$(echo "$MEJOR" | cut -d'+' -f1 | xargs).pt"
E="models/candidatos/$(echo "$MEJOR" | cut -d'+' -f2 | xargs).pt"
echo "=== $(date +%H:%M) mejor: $G + $E"
$PY -m src.embedding.construir_ensamble --generalista "$G" --especialista "$E" > $L/log_construir.txt 2>&1 || { echo FALLO construir; exit 1; }
echo "=== $(date +%H:%M) evaluación final"
$PY -m src.embedding.evaluar_ensamble > $L/log_evaluacion_final.txt 2>&1 || { echo FALLO evaluacion; exit 1; }
$PY -m unittest serving.tests.test_ensamble serving.tests.test_gallery_store > $L/log_pruebas.txt 2>&1 && echo "pruebas ok" || echo "FALLO pruebas"
echo "=== $(date +%H:%M) FINAL TERMINADO"
