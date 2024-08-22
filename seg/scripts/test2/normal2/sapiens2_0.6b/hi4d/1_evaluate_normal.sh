#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../../.. || exit

# Set the configurations for detection and pose estimation
MODE='hi4d'
DATASET='normal_render_people'
MODEL="sapiens2_0.6b_${DATASET}-4096x3072"

RUN_FILE=tools/normal/test_${MODE}_normal.py
DATA_DIR=/home/rawalk/Desktop/sapiens/seg/data/${MODE}/evaluation ## ground_truth
PRED_DIR=/home/rawalk/Desktop/sapiens/seg/Outputs/test/${DATASET}/${MODEL}/${MODE}/evaluation

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}"
