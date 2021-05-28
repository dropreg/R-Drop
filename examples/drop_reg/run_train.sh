#!/bin/bash
src=de
tgt=en

gpu_device=0

data_dir=/data/$src-$tgt-dir/
save_dir=/data/$src-$tgt-ckpt/
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=$gpu_device python3 ../../fairseq_cli/train.py $data_dir \
    --user-dir examples/drop_reg/drop_reg_src \
    --task drop_reg_translation \
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
    --save-dir $save_dir | tee -a $save_dir/train.log \
