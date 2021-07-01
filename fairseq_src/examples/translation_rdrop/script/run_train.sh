#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

src=de
tgt=en

DATA_PATH=data-bin/iwslt14.rdrop.tokenized.de-en/
MODEL_PATH=iwslt14.rdrop.de-en-ckpt
mkdir -p $MODEL_PATH
nvidia-smi

python -c "import torch; print(torch.__version__)"

export CUDA_VISIBLE_DEVICES=0

fairseq-train $DATA_PATH \
    --user-dir examples/translation_rdrop/translation_rdrop_src/ \
    --task rdrop_translation \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --lr 0.0005 -s $src -t $tgt \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion reg_label_smoothed_cross_entropy \
    --reg-alpha 5 \
    --no-progress-bar \
    --seed 64 \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 300000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    --save-dir $MODEL_PATH | tee -a $MODEL_PATH/train.log \
