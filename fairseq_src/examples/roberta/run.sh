

TOTAL_NUM_UPDATES=33112
WARMUP_UPDATES=1986
LR=1e-5
NUM_CLASSES=2
MAX_SENTENCES=8
UPDATE_DREQ=4
ROBERTA_PATH=/data/roberta.large/model.pt
TASK=QNLI
DATA_DIR=/data/result/$TASK-bin

save_dir=/data/roberta/$TASK-mean-5
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=7 fairseq-train $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --update-freq $UPDATE_DREQ \
    --max-tokens 4400 \
    --task reg_sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --dropout 0.1 --attention-dropout 0.1 \
    --criterion reg_sentence_prediction \
    --num-classes $NUM_CLASSES \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --no-progress-bar \
    --seed 1 \
    --regression-target --best-checkpoint-metric loss \
    --no-save \
    --save-dir $save_dir | tee -a $save_dir/train.log \
