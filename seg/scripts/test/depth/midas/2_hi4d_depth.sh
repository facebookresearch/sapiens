#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='tools/depth/test_hi4d_depth.py'


## midas large
DATA_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/data/hi4d/evaluation' ## ground_truth
PRED_DIR='/mnt/home/rawalk/drive/seg/Outputs/test/midas/hi4d/evaluation'

# ## midas swin
# DATA_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/data/hi4d/evaluation' ## ground_truth
# PRED_DIR='/mnt/home/rawalk/drive/seg/Outputs/test/midas_swin/hi4d/evaluation'

####--------------------------v--------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}" 
