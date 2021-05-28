# fairseq-preprocess  \
#     --source-lang en --target-lang de \
#     --trainpref /data/lxb/wmt_data/wmt16_en_de/train.tok.clean.bpe.32000 \
#     --validpref /data/lxb/wmt_data/wmt16_en_de/newstest2013.tok.bpe.32000 \
#     --testpref /data/lxb/wmt_data/wmt16_en_de/newstest2014.tok.bpe.32000 \
#     --destdir /data/lxb/wmt_data/wmt16_en_de/wmt16_en_de_bpe32k \
#     --nwordssrc 32768 --nwordstgt 32768 \
#     --joined-dictionary \
#     --workers 20


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train /data/lxb/wmt_data/wmt16_en_de/wmt16_en_de_bpe32k \
#     --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#     -s en -t de \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.0005 \
#     --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
#     --dropout 0.3 --weight-decay 0.0 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --save-dir /data/lxb/nmt_checkpoint/wmt_checkpoint/16_en_de_scale_baseline/ \
#     --no-progress-bar \
#     --fp16


# python ../../scripts/average_checkpoints.py \
#     --inputs /data/lxb/nmt_checkpoint/wmt_checkpoint/16_en_de_scale_baseline/ \
#     --num-epoch-checkpoints 10 \
#     --output /data/lxb/nmt_checkpoint/wmt_checkpoint/16_en_de_scale_baseline/checkpoint.avg10.pt


CUDA_VISIBLE_DEVICES=7 fairseq-generate /data/lxb/wmt_data/wmt16_en_de/wmt16_en_de_bpe32k \
    --path /data/lxb/nmt_checkpoint/wmt_checkpoint/16_en_de_scale_baseline/checkpoint.avg10.pt \
    --beam 4 --lenpen 0.6 --remove-bpe --quiet \


# fairseq-generate /data/lxb/wmt_data/wmt16_en_de/wmt16_en_de_bpe32k \
#     --path /data/lxb/nmt_checkpoint/wmt_checkpoint/16_en_de_scale_baseline/checkpoint.avg10.pt \
#     --beam 4 --lenpen 0.6 --remove-bpe > gen.out

# bash ../../scripts/compound_split_bleu.sh gen.out
# bash ../../scripts/sacrebleu.sh wmt14/full en de gen.out