#!/bin/bash

cd ../../.. || exit

SAPIENS_CHECKPOINT_ROOT=/home/${USER}/sapiens_host
OUTPUT_CHECKPOINT_ROOT=/home/${USER}/sapiens_lite_host

MODE='torchscript' ## original. no optimizations (slow). full precision inference.
# MODE='bfloat16' ## A100 gpus. faster inference at bfloat16
# MODE='float16' ## V100 gpus. faster inference at float16 (no flash attn)

OUTPUT_CHECKPOINT_ROOT=$OUTPUT_CHECKPOINT_ROOT/$MODE

VALID_GPU_IDS=(3)

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194.pth
MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178.pth
# MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_2b/sapiens_2b_goliath_best_goliath_mIoU_8111_epoch_155.pth

OUTPUT_CHECKPOINT_PATH=${OUTPUT_CHECKPOINT_ROOT}/seg/checkpoints/${MODEL_NAME}/$(basename ${CHECKPOINT%.pth}_$MODE.pt2)

DATASET='goliath'
MODEL="${MODEL_NAME}_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_seg/${DATASET}/${MODEL}.py"

CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} python tools/deployment/pytorch2torchscript.py ${CONFIG_FILE} \
        --checkpoint ${CHECKPOINT} \
        --output-file ${OUTPUT_CHECKPOINT_PATH} \
        --shape 1024 768 ## height, width
