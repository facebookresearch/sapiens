#!/bin/bash

cd ../.. || exit

#----------------------------set your input and output directories----------------------------------------------
GLOBAL_START_INDEX=$1
GLOBAL_END_INDEX=$2

JOBS_PER_GPU=8; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
RUN_FILE='process/pose_video_process.py'

#--------------------------MODEL CARD---------------
MODEL_NAME='sapiens_1b'; CHECKPOINT='/uca/rawalk/sapiens_host/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth'
DATASET='coco_wholebody'
MODEL="${MODEL_NAME}-210e_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_pose/${DATASET}/${MODEL}.py"

# bounding box detector
DETECTION_CONFIG_FILE='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DETECTION_CHECKPOINT='/uca/rawalk/sapiens_host/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

##-------------------------------------inference-------------------------------------
TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))
RANGE_PER_JOB=$(( (GLOBAL_END_INDEX - GLOBAL_START_INDEX) / TOTAL_JOBS ))

RANGE_PER_JOB=$(( RANGE_PER_JOB > 0 ? RANGE_PER_JOB : 1 ))

for ((i=0; i<TOTAL_JOBS; i++));
do

    START_INDEX=$(( GLOBAL_START_INDEX + i * RANGE_PER_JOB ))
    END_INDEX=$(( START_INDEX + RANGE_PER_JOB ))

    # Make sure the last job covers the remaining range
    if [[ $i -eq $((TOTAL_JOBS - 1)) ]]; then
        END_INDEX=$GLOBAL_END_INDEX
    fi

    # Check if END_INDEX is greater than GLOBAL_END_INDEX
    if [[ $END_INDEX -gt $GLOBAL_END_INDEX ]]; then
        break
    fi

    GPU_ID=$((i % TOTAL_GPUS))
    CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} python ${RUN_FILE} \
        ${DETECTION_CONFIG_FILE} \
        ${DETECTION_CHECKPOINT} \
        ${CONFIG_FILE} \
        ${CHECKPOINT} \
        --start_index ${START_INDEX} \
        --end_index ${END_INDEX} &

    sleep 1
done


# Wait for all background processes to finish
wait
