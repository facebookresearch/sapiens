#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=/mnt/home/rawalk/sapiens_host
OUTPUT_CHECKPOINT_ROOT=/mnt/home/rawalk/sapiens_lite_host

# MODE='torchscript' ## original. no optimizations (slow). full precision inference.
# MODE='bfloat16' ## A100 gpus. faster inference at bfloat16
MODE='float16' ## V100 gpus. faster inference at float16 (no flash attn)

OUTPUT_CHECKPOINT_ROOT=$OUTPUT_CHECKPOINT_ROOT/$MODE

VALID_GPU_IDS=(3)

#--------------------------MODEL CARD---------------
MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_shutterstock_instagram_epoch_1600.pth
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_0.6b/sapiens_0.6b_shutterstock_instagram_epoch_1600.pth
# MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_1b/sapiens_1b_shutterstock_instagram_epoch_173.pth
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_2b/sapiens_2b_shutterstock_instagram_epoch_660.pth

OUTPUT=${OUTPUT_CHECKPOINT_ROOT}/pretrain/checkpoints/${MODEL_NAME}/

DATASET='shutterstock_instagram_test' ## for feature extraction
MODEL="mae_${MODEL_NAME}-p16_8xb512-coslr-1600e_${DATASET}"
CONFIG_FILE="configs/sapiens_mae/${DATASET}/${MODEL}.py"

BATCH_SIZE=6

TORCHDYNAMO_VERBOSE=1  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} \
            python3 tools/deployment/torch_optimization.py ${CONFIG_FILE} ${CHECKPOINT} --output-dir ${OUTPUT} --fp16 --max-batch-size ${BATCH_SIZE}