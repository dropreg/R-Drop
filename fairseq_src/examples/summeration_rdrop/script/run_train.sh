#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=16
DATA_PATH=cnn_dm-bin/
BART_PATH=bart.large/model.pt
MODEL_PATH=/data/lxb/test/bart-large-checkpoints/
mkdir -p $MODEL_PATH
nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3 

fairseq-train $DATA_PATH \
    --user-dir examples/summeration_rdrop/summeration_rdrop_src \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task rdrop_summeration \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --reg-alpha 0.7 \
    --criterion reg_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --no-progress-bar \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --save-dir $MODEL_PATH | tee -a $MODEL_PATH/train.log \