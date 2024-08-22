#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=/home/${USER}/sapiens_host
OUTPUT_CHECKPOINT_ROOT=/home/${USER}/sapiens_lite_host

# MODE='float16' ### no flash attention. for V100 gpus. default
MODE='bfloat16' ## for A100 gpus. better performance.

OUTPUT_CHECKPOINT_ROOT=$OUTPUT_CHECKPOINT_ROOT/$MODE

VALID_GPU_IDS=(3)

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194.pth
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178.pth
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_2b/sapiens_2b_goliath_best_goliath_mIoU_8111_epoch_155.pth

OUTPUT=${OUTPUT_CHECKPOINT_ROOT}/seg/checkpoints/${MODEL_NAME}/

DATASET='goliath'
MODEL="${MODEL_NAME}_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_seg/${DATASET}/${MODEL}.py"

TORCHDYNAMO_VERBOSE=1  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} python3 tools/deployment/torch_optimization.py ${CONFIG_FILE} ${CHECKPOINT} --output-dir ${OUTPUT}
