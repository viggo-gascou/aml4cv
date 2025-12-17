#!/usr/bin/env bash

BATCH_SIZE=64
AUGMENTATION_PROBA=0.3
EPOCHS=100
# fine-tune pre-trained
uv run train --batch-size $BATCH_SIZE --model pretrained --epochs $EPOCHS --patience $EPOCHS --augmentation-proba $AUGMENTATION_PROBA --learning-rate 0.00005 --early-stop-criterion val/loss


# train baseline (from scratch model)
uv run train --batch-size $BATCH_SIZE --model base --epochs $EPOCHS --patience $EPOCHS --augmentation-proba $AUGMENTATION_PROBA --learning-rate 0.0001 --early-stop-criterion val/loss
