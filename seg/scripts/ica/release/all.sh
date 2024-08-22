#!/bin/bash

INPUT_ROOT_DIR='/home/rawalk/Desktop/sapiens/seg/data/shutterstock_videos'
OUTPUT_ROOT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/shutterstock_videos'

# VIDEOS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
VIDEOS=(02 03 04 05 06 07 08)
# VIDEOS=(09 10 11 12 13 14 15 16)
# VIDEOS=(17 18 19 20 21 22 23 24)
# VIDEOS=(25 26 27 28 29 30 31)

JOBS_PER_GPU=4
VALID_GPU_IDS="0 1 2 3 4 5 6 7" # total jobs = jobs per gpu x number of gpus

for VIDEO in "${VIDEOS[@]}"; do
    INPUT_DIR="$INPUT_ROOT_DIR/$VIDEO"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/$VIDEO"

    ##------------------------------------------------------------
    ./0_pose.sh "$INPUT_DIR" "$OUTPUT_DIR" $JOBS_PER_GPU "$VALID_GPU_IDS"
    wait

    ./1_seg.sh "$INPUT_DIR" "$OUTPUT_DIR" $JOBS_PER_GPU "$VALID_GPU_IDS"
    wait

    ./2_depth.sh "$INPUT_DIR" "$OUTPUT_DIR" $JOBS_PER_GPU "$VALID_GPU_IDS"
    wait

    ./3_normal.sh "$INPUT_DIR" "$OUTPUT_DIR" $JOBS_PER_GPU "$VALID_GPU_IDS"
    wait

    ./4_make_video.sh "$OUTPUT_DIR"
done
