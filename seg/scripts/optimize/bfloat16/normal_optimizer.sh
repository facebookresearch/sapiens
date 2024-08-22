#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=/home/${USER}/sapiens_host
OUTPUT_CHECKPOINT_ROOT=/home/${USER}/sapiens_lite_host

# MODE='float16' ### no flash attention. for V100 gpus. default
MODE='bfloat16' ## for A100 gpus. better performance.

OUTPUT_CHECKPOINT_ROOT=$OUTPUT_CHECKPOINT_ROOT/$MODE

VALID_GPU_IDS=(3)
#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.3b/sapiens_0.3b_normal_render_people_epoch_66.pth
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.6b/sapiens_0.6b_normal_render_people_epoch_200.pth
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115.pth
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70.pth

OUTPUT=${OUTPUT_CHECKPOINT_ROOT}/normal/checkpoints/${MODEL_NAME}/

DATASET='normal_render_people'
MODEL="${MODEL_NAME}_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_normal/${DATASET}/${MODEL}.py"

TORCHDYNAMO_VERBOSE=1  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} python3 tools/deployment/torch_optimization.py ${CONFIG_FILE} ${CHECKPOINT} --output-dir ${OUTPUT}
