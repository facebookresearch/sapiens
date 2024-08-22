#!/bin/bash

cd ../.. || exit

#----------------------------set your input and output directories----------------------------------------------
# START_INDEX=0
# END_INDEX=6000

START_INDEX=0
END_INDEX=16

#--------------------------MODEL CARD---------------
MODEL_NAME='sapiens_1b'; CHECKPOINT='/uca/rawalk/sapiens_host/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth'
DATASET='coco_wholebody'
MODEL="${MODEL_NAME}-210e_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_pose/${DATASET}/${MODEL}.py"

# bounding box detector
DETECTION_CONFIG_FILE='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DETECTION_CHECKPOINT='/uca/rawalk/sapiens_host/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

##-------------------------------------inference-------------------------------------
RUN_FILE='process/pose_video_process.py'

##---------with visualization flag-----------
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python ${RUN_FILE} \
#     ${DETECTION_CONFIG_FILE} \
#     ${DETECTION_CHECKPOINT} \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --start_index ${START_INDEX} \
#     --end_index ${END_INDEX} \
#     --visualize \

##------------no visualization flag-----------------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, python ${RUN_FILE} \
    ${DETECTION_CONFIG_FILE} \
    ${DETECTION_CHECKPOINT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --start_index ${START_INDEX} \
    --end_index ${END_INDEX} \
