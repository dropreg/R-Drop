#!/usr/bin/env bash
set -x
set -e
#---------------------------------------
ARCH=${1:-"transformer_lm_gpt"}
PREFIX=~/

DATA=wikitext-103
SEED=1
DATA_PATH=$PREFIX/data/lm_data/${DATA}_bin
nvidia-smi


MODEL_PATH=$PREFIX/models/lm_self_reg_${ARCH}_${DATA}_${KL_WEIGHT}
mkdir -p ${MODEL_PATH}

python -c "import torch; print(torch.__version__)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

fairseq-train $DATA_PATH \
  --ddp-backend=no_c10d \
  --arch $ARCH --share-decoder-input-output-embed \
  --save-dir $MODEL_PATH \
  --user-dir /examples/language_model_rdrop/language_model_rdrop_src \
  --dropout 0.1 \
  --task rdrop_lm \
  --criterion reg_cross_entropy \
  --reg-alpha 1.0 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2000 --update-freq 16 \
  --fp16 \
  --max-update 50000 --skip-invalid-size-inputs-valid-test \
  --tensorboard-logdir $MODEL_PATH/tensorboard 


