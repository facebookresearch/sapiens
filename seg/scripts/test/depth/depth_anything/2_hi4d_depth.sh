#!/bin/bash

MODEL_NAME='vitl'
# MODEL_NAME='vitb'
# MODEL_NAME='vits'

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='tools/depth/test_hi4d_depth.py'

## depth anythin
DATA_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/data/hi4d/evaluation' ## ground_truth
PRED_DIR=/mnt/home/rawalk/drive/seg/Outputs/test/depth_anything_$MODEL_NAME/hi4d/evaluation

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}" 
