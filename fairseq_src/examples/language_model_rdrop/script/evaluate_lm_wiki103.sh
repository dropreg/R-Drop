#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

DATA=wikitext-103
DATA_PATH=$PREFIX/${DATA}/data-bin

MODEL_PATH=$PREFIX/lm_${ARCH}_${DATA}_ckpt
mkdir -p ${MODEL_PATH}
nvidia-smi

python -c "import torch; print(torch.__version__)"

export CUDA_VISIBLE_DEVICES=0

fairseq-eval-lm $DATA_PATH \
    --path $MODEL_PATH/checkpoint_best.pt \
    --max-sentences 2 \
    --tokens-per-sample 512 \
    --context-window 400
