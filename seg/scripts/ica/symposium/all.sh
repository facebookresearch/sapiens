#!/bin/bash

ROOT_DIR='/home/rawalk/Desktop/sapiens/seg/data/full_head_ICA'

IMAGE_DIR='body'
# IMAGE_DIR='glasses'
# IMAGE_DIR='headrom'
# IMAGE_DIR='neutralscan'

ICA_DIR=$ROOT_DIR/$IMAGE_DIR

JOBS_PER_GPU=4; VALID_GPU_IDS="0 1 2 3 4 5 6 7" ## total jobs = jobs per gpu x number of gpus

##------------------------------------------------------------
# ./0_pose.sh "$ICA_DIR" "$JOBS_PER_GPU" "$VALID_GPU_IDS" &
# wait

# ./1_seg.sh "$ICA_DIR" "$JOBS_PER_GPU" "$VALID_GPU_IDS" &
# wait

# ./2_depth.sh "$ICA_DIR" "$JOBS_PER_GPU" "$VALID_GPU_IDS" &
# wait

# ./3_normal.sh "$ICA_DIR" "$JOBS_PER_GPU" "$VALID_GPU_IDS" &
# wait

# ./4_albedo_render_people.sh "$ICA_DIR" "$JOBS_PER_GPU" "$VALID_GPU_IDS" &
# wait


./5_make_video.sh $ICA_DIR
