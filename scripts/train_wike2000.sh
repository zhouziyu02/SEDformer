#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATASET="Wike2000_missing_25pct"
BATCH_SIZE=16
HISTORY=90
PRED_LEN=30
PATIENCE=50
LR=0.001
EPOCHS=200
GPU=0

export SED_DIM=32
export SED_LIF_TAU=3
export SED_LAYERS=2
export SED_POOL=4

cd "$(dirname "$0")/.."
python train.py \
    --dataset "$DATASET" \
    --history "$HISTORY" \
    --pred_len "$PRED_LEN" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed 1 \
    --gpu "$GPU" \
    --epoch "$EPOCHS"
