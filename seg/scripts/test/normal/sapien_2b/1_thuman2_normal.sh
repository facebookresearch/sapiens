#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='tools/normal/test_thuman2_normal.py'

# ## sapiens
DATA_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/data/thuman2/evaluation' ## ground_truth
PRED_DIR='/mnt/home/rawalk/Desktop/sapiens/seg/Outputs/test/normal_render_people/sapien_2b_normal_render_people-1024x768/thuman2/evaluation'

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DEVICES=0,1,2,3,4,5,6,

CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
  --data_dir ${DATA_DIR} \
  --pred_dir="${PRED_DIR}" 
