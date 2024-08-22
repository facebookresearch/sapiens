#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# MODEL_NAME='vitl'
MODEL_NAME='vitb'

# Set the configurations for detection and pose estimation
RUN_FILE='tools/depth/test_thuman2_depth.py'

## depth anything
DATA_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/data/thuman2/evaluation' ## ground_truth
PRED_DIR=/mnt/home/rawalk/drive/seg/Outputs/test/depth_anything_$MODEL_NAME/thuman2/evaluation


####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}" 
