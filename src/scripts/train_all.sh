#!/usr/bin/env bash

BATCH_SIZE=64
AUGMENTATION_PROBA=0.3
# fine-tune pre-trained
uv run train --batch-size $BATCH_SIZE --model pretrained --epochs 30 --augmentation-proba $AUGMENTATION_PROBA --learning-rate 0.00005

# train baseline (from scratch model)
uv run train --batch-size $BATCH_SIZE --model baseline --epochs 100 --patience 100 --augmentation-proba $AUGMENTATION_PROBA --learning-rate 0.0001
