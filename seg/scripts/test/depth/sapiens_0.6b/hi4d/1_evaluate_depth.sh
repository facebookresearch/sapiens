#!/bin/bash

cd ../../../../.. || exit

# Set the configurations for detection and pose estimation
MODE='hi4d'
DATASET='render_people'

MODEL="sapiens_0.6b_${DATASET}-1024x768"

RUN_FILE=tools/depth/test_${MODE}_depth.py
DATA_DIR=/home/rawalk/Desktop/sapiens/seg/data/${MODE}/evaluation ## ground_truth
PRED_DIR=/home/rawalk/Desktop/sapiens/seg/Outputs/test/${DATASET}/${MODEL}/${MODE}/evaluation

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}"
