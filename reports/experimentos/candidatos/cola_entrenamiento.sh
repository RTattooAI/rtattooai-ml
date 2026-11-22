#!/usr/bin/env bash
# Cola de candidatos entrenados SIN el conjunto de validación (identidades_train_sinval.json).
cd "$(dirname "$0")/../../.."
PY=venv/Scripts/python.exe
M=datasets/real_crops/identidades_train_sinval.json
L=reports/experimentos/candidatos
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf8
run() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "$L/log_$NAME.txt" 2>&1 && echo "ok $NAME" || echo "FALLO $NAME"; }
NAME=head_ref;      run "$NAME" $PY -m src.embedding.finetune_head --manifiesto $M --k-real 6  --max-degrade 0.5 --epochs 60  --salida models/candidatos/head_ref.pt
NAME=head_k10;      run "$NAME" $PY -m src.embedding.finetune_head --manifiesto $M --k-real 10 --max-degrade 0.5 --epochs 100 --salida models/candidatos/head_k10.pt
NAME=head_k10_h1024; run "$NAME" $PY -m src.embedding.finetune_head --manifiesto $M --k-real 10 --max-degrade 0.5 --epochs 100 --hidden 1024 --salida models/candidatos/head_k10_h1024.pt
NAME=cnn_r18_ref;   run "$NAME" $PY -m src.embedding.train_cnn --pesos imagenet --manifiesto $M --k-real 4 --max-degrade 0.5 --epochs 30 --salida models/candidatos/cnn_r18_ref.pt
NAME=cnn_r18_k6e50; run "$NAME" $PY -m src.embedding.train_cnn --pesos imagenet --manifiesto $M --k-real 6 --max-degrade 0.5 --epochs 50 --salida models/candidatos/cnn_r18_k6e50.pt
NAME=cnn_r50;       run "$NAME" $PY -m src.embedding.train_cnn --pesos imagenet --arquitectura resnet50 --ids-per-batch 48 --manifiesto $M --k-real 4 --max-degrade 0.5 --epochs 30 --salida models/candidatos/cnn_r50.pt
echo "=== $(date +%H:%M) COLA TERMINADA"
