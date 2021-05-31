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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

fairseq-train $DATA_PATH \
  --task language_modeling \
  --arch transformer_lm_gpt --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2000 --update-freq 16 \
  --fp16 --ddp-backend=no_c10d \
  --max-update 50000 --skip-invalid-size-inputs-valid-test \
  --tensorboard-logdir $MODEL_PATH/tensorboard \
  --save-dir $MODEL_PATH | tee -a $MODEL_PATH/train.log \
