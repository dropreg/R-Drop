#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

src=de
tgt=en

DATA_PATH=data-bin/iwslt14.rdrop.tokenized.de-en/
MODEL_PATH=iwslt14.rdrop.de-en-ckpt
nvidia-smi

python -c "import torch; print(torch.__version__)"

export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/generate.py $DATA_PATH \
    --path $MODEL_PATH/checkpoint_best.pt \
    --beam 5 --remove-bpe >> $MODEL_PATH/result.gen

bash scripts/compound_split_bleu.sh $MODEL_PATH/result.gen
