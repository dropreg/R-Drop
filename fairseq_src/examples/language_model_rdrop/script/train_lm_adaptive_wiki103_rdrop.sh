#!/usr/bin/env bash
set -x
set -e
#---------------------------------------
ARCH=${1:-"transformer_lm_wiki103"}
PREFIX=~/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA=wikitext-103
SEED=1
LR=1e-4
DATA_PATH=$PREFIX/data/lm_data/${DATA}_bin
CODE_PATH=$PREFIX/fariseq
nvidia-smi

MODEL_PATH=$PREFIX/models/lm_self_reg_ada_${ARCH}_${DATA}
mkdir -p ${MODEL_PATH}

python -c "import torch; print(torch.__version__)"

python ${CODE_PATH}/train.py ${DATA_PATH} \
  --task uncertainty_lm \
  --user-dir /examples/language_model_rdrop/language_model_rdrop_src \
  --save-dir ${MODEL_PATH} --arch ${ARCH} \
  --max-update 320000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 \
  --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
  --optimizer nag --min-lr ${LR} --clip-norm 0.1 \
  --criterion reg_adaptive_loss \
  --reg-alpha 1.0 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed ${SEED} \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test \
  --ddp-backend=legacy_ddp \
  --fp16 \
  --restore-file ${MODEL_PATH}/checkpoint_best.pt \
  --tensorboard-logdir $MODEL_PATH/tensorboard

