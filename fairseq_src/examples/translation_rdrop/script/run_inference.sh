#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

src=de
tgt=en

DATA_PATH=iwslt14.tokenized.de-en/data-bin/
MODEL_PATH=iwslt14.tokenized.de-en-ckpt/
nvidia-smi

python -c "import torch; print(torch.__version__)"

export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/generate.py $DATA_PATH \
    --path $MODEL_PATH/checkpoint_best.pt \
    --beam 5 --remove-bpe --lenpen 0.5 >> $MODEL_PATH/result.gen

bash ../../scripts/compound_split_bleu.sh $MODEL_PATH/result.gen
