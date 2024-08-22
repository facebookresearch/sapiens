#!/bin/bash

cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='tools/normal/test_hi4d_normal.py'

## depth anythin
DATA_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/data/hi4d/evaluation' ## ground_truth
PRED_DIR=/mnt/home/rawalk/drive/seg/Outputs/test/pifuhd/hi4d/evaluation

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}" 
