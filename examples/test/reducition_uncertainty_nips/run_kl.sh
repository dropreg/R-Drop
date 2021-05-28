#!/bin/bash
train_flag=false
keep_dir=false
gpu_device=0 #,1,2,3

src=de
tgt=en

# de es fr zh
if [ $src == de ] || [ $tgt == de ]; then
    data_dir=/data/lxb/iwslt_data/de-en_file/databin/
elif [ $src == es ] || [ $tgt == es ]; then
    data_dir=/data/lxb/iwslt_data/es-en_file/databin/
elif [ $src == fr ] || [ $tgt == fr ]; then
    data_dir=/data/lxb/iwslt_data/fr-en_file/databin/
else
    data_dir=/data/lxb/iwslt_data/zh-en_file/databin/
fi

# dropout
save_dir=/data/lxb/nmt_checkpoint/iwslt_checkpoint/nips/$src-$tgt-ckpt/$src-$tgt-ld-ro-01-baseline

if [ $train_flag == true ]; then
    echo "train nmt model from lang $src to $tgt "
    mkdir -p $save_dir
    if [ $keep_dir != true ]; then
        rm -rf $save_dir
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu_device fairseq-train $data_dir \
        --ddp-backend=c10d \
        --user-dir examples/test/reducition_uncertainty_nips/uncertainty \
        --task uncertainty_translation \
        --arch variants_transformer \
        --share-all-embeddings \
        --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 2048 \
        --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion uncertainty_label_smoothed_cross_entropy \
        --decoder-layerdrop 0.1 \
        --no-progress-bar \
        --seed 64 \
        --fp16 \
        --max-epoch 500 --warmup-updates 6000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
        --save-dir $save_dir | tee -a $save_dir/train.log \

    # CUDA_VISIBLE_DEVICES=$gpu_device fairseq-train $data_dir \
    #     --user-dir examples/reducition_uncertainty_nips/uncertainty \
    #     --task uncertainty_translation \
    #     --arch variants_transformer \
    #     --share-all-embeddings \
    #     --optimizer adam --lr 0.0005 -s $src -t $tgt \
    #     --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    #     --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    #     --criterion uncertainty_label_smoothed_cross_entropy \
    #     --no-progress-bar \
    #     --seed 64 \
    #     --fp16 \
    #     --max-update 300000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    #     --save-dir $save_dir | tee -a $save_dir/train.log \

else

    for i in {200..500}
    do

        echo "test nmt model from lang $src to $tgt for chekpoint $i" >> ld-ro.bleu
        CUDA_VISIBLE_DEVICES=$gpu_device fairseq-generate  $data_dir \
            --user-dir examples/test/reducition_uncertainty_nips/uncertainty \
            --task uncertainty_translation \
            -s $src -t $tgt \
            --path $save_dir/checkpoint${i}.pt \
            --batch-size 128 --beam 5 --remove-bpe --quiet | tail -1 >> ld-ro.bleu

    done
    exit

    if [ $src == de ] || [ $tgt == de ]; then

        CUDA_VISIBLE_DEVICES=$gpu_device fairseq-generate  $data_dir \
            -s $src -t $tgt \
            --user-dir examples/test/reducition_uncertainty_nips/uncertainty \
            --task uncertainty_translation \
            --decoder-layers-to-keep 0 \
            --path $save_dir/checkpoint_best.pt \
            --batch-size 128 --beam 5 --remove-bpe --quiet \
    
    elif [ $src == es ] || [ $tgt == es ]; then
        
        bpe_in=/data/lxb/iwslt_data/es-en_file/test.$src
        bpe_ref=/data/lxb/iwslt_data/es-en_file/test.$tgt.debpe.detok
        cat $bpe_in | CUDA_VISIBLE_DEVICES=$gpu_device fairseq-interactive $data_dir \
            --user-dir examples/reducition_uncertainty_nips/uncertainty \
            --task uncertainty_translation \
            --source-lang $src --target-lang $tgt \
            --path $save_dir/checkpoint_best.pt \
            --buffer-size 2000 --batch-size 128 \
            --beam 5 --remove-bpe > iwslt17.test.$src-$tgt.$tgt.sys
        grep ^H iwslt17.test.$src-$tgt.$tgt.sys | cut -f3 > out.log
        sacremoses -l $tgt detokenize < out.log > deout.log
        cat deout.log | sacrebleu $bpe_ref --language-pair $src-$tgt
        rm iwslt17.test.$src-$tgt.$tgt.sys
        rm out.log
        rm deout.log

    elif [ $src == fr ] || [ $tgt == fr ]; then
        
        bpe_in=/data/lxb/iwslt_data/fr-en_file/test.$src
        cat $bpe_in | CUDA_VISIBLE_DEVICES=$gpu_device fairseq-interactive $data_dir \
            --user-dir examples/teacher_distillation/teac_distill \
            --task reorderd_teacher_translation \
            --valid-decoder-order-idx 0 \
            --source-lang $src --target-lang $tgt \
            --path $save_dir/checkpoint_best.pt \
            --buffer-size 2000 --batch-size 128 \
            --beam 5 --remove-bpe > iwslt17.test.$src-$tgt.$tgt.sys
        grep ^H iwslt17.test.$src-$tgt.$tgt.sys | cut -f3 > out.log
        sacremoses -l $tgt detokenize < out.log > deout.log
        cat deout.log | sacrebleu --test-set iwslt17 --language-pair $src-$tgt

        rm iwslt17.test.$src-$tgt.$tgt.sys
        rm out.log
        rm deout.log

    else

        bpe_in=/data/lxb/iwslt_data/zh-en_file/test.$src
        cat $bpe_in | CUDA_VISIBLE_DEVICES=$gpu_device fairseq-interactive $data_dir \
            --user-dir examples/teacher_distillation/teac_distill \
            --task reorderd_teacher_translation \
            --valid-decoder-order-idx 0 \
            --source-lang $src --target-lang $tgt \
            --path $save_dir/checkpoint_best.pt \
            --buffer-size 2000 --batch-size 128 \
            --beam 5 --remove-bpe > iwslt17.test.$src-$tgt.$tgt.sys
        grep ^H iwslt17.test.$src-$tgt.$tgt.sys | cut -f3 > out.log
        sacremoses -l $tgt detokenize < out.log > deout.log
        cat deout.log | sacrebleu --test-set iwslt17 --language-pair $src-$tgt

        rm iwslt17.test.$src-$tgt.$tgt.sys
        rm out.log
        rm deout.log

    fi

fi
