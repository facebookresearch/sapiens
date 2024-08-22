#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/demo_pose3d_vis.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/pose/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/pose/Outputs/vis'

####------------------------------------------
DATASET='goliath3d'
MODEL="sapiens_1b_${DATASET}"

CONFIG_FILE="configs/sapiens_pose3d/${DATASET}/${MODEL}.py"
# CHECKPOINT='/home/rawalk/drive/pose/Outputs/train/goliath3d/sapiens_1b_goliath3d/slurm/08-09-2024_18:50:09/iter_45000.pth'
CHECKPOINT='/home/rawalk/drive/pose/Outputs/train/goliath3d/sapiens_1b_goliath3d/slurm/08-09-2024_18:50:09/iter_55000.pth'

# # # # # # # ##------------------------------data:in the wild---------------------------------------
DATA_PREFIX="nvidia_in_the_wild/69000"
# DATA_PREFIX="nvidia_in_the_wild/69000_small"
# DATA_PREFIX="nvidia_in_the_wild/69000_tiny"

XY_AXIS_LIMIT=1.8; Z_AXIS_LIMIT=2.0

LINE_THICKNESS=20
RADIUS=20
KPT_THRES=0.5 ## default keypoint confidence

# # # # # # # ###------------------------------data:goliath--------------------------
# DATA_DIR='/home/rawalk/Desktop/sapiens/pose/data/goliath/test/images/s--20190524--1430--4911137--GHS'
# # DATA_PREFIX='400178'; SKIP_FACE=False; XY_AXIS_LIMIT=1.8; Z_AXIS_LIMIT=2.0
# DATA_PREFIX='400178'; SKIP_FACE=False; XY_AXIS_LIMIT=0.8; Z_AXIS_LIMIT=1.0

# LINE_THICKNESS=20
# RADIUS=10
# KPT_THRES=0.5

# # ###---------------------------------data ica--------------------------------------
# DATA_DIR='/uca/full_head_ICA/2022-10-19/Chen/Segs/BodySpin/image'
# DATA_PREFIX='iPhone_rgb'; SKIP_FACE=False; XY_AXIS_LIMIT=0.1; Z_AXIS_LIMIT=5.0

# LINE_THICKNESS=20
# RADIUS=10
# KPT_THRES=0.98

# # # # # ##------------------------------data:itw videos---------------------------------------
# DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
# # # DATA_PREFIX="itw_videos/reel01" ## skipping girl in pink, low res
# # # DATA_PREFIX="itw_videos/reel02" ## skipping guy, low res
# # # DATA_PREFIX="itw_videos/reel03" ## skipping girl in black, high res
# # # DATA_PREFIX="itw_videos/reel04" ## curly hair girl
# # # DATA_PREFIX="itw_videos/reel05" ## birthday cake
# # # DATA_PREFIX="itw_videos/reel06" ## side profile, girl
# # # DATA_PREFIX="itw_videos/reel07" ## yellow bedroom
# # # DATA_PREFIX="itw_videos/reel08" ## two girls, standing on bed
# # # DATA_PREFIX="itw_videos/reel09" ## chess video
# # # DATA_PREFIX="itw_videos/reel10" ## face wash video
# # # DATA_PREFIX="itw_videos/reel40" ## guy in the cap by water
# # # DATA_PREFIX="itw_videos/reel36"
# # # DATA_PREFIX="itw_videos/reel37"
# # # DATA_PREFIX="itw_videos/reel35"
# # # DATA_PREFIX="itw_videos/reel22"
# # # DATA_PREFIX="itw_videos/reel38"
# # # DATA_PREFIX="itw_videos/reel35"
# # # DATA_PREFIX="itw_videos/reel30"
# # # DATA_PREFIX="itw_videos/reel09"

# # ## the main video exercising
# DATA_PREFIX="itw_videos/reel11"; XY_AXIS_LIMIT=1.8; Z_AXIS_LIMIT=2.0
# # DATA_PREFIX="itw_videos/reel40"; XY_AXIS_LIMIT=1.8; Z_AXIS_LIMIT=2.0
# # DATA_PREFIX="itw_videos/reel10"; XY_AXIS_LIMIT=1.8; Z_AXIS_LIMIT=2.0

# LINE_THICKNESS=20
# # RADIUS=20
# RADIUS=10

# # KPT_THRES=0.3 ## low thresholder
# KPT_THRES=0.5 ## medium thresholder
# # KPT_THRES=0.7 ## higher thresholder

##------------------------------data:shutterstock---------------------------------------
# DATA_DIR='/home/rawalk/Desktop/sapiens/pretrain/data'
# DATA_PREFIX='shutterstock'

# LINE_THICKNESS=4
# RADIUS=3
# KPT_THRES=0.3

##------------------------------------------------------------------------------
INPUT="${DATA_DIR}/${DATA_PREFIX}"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}" .pth)
OUTPUT="${OUTPUT_DIR}/${DATASET}/${MODEL}/${DATA_PREFIX}/${CHECKPOINT_NAME}_output"

JOBS_PER_GPU=4; TOTAL_GPUS=8
# JOBS_PER_GPU=1; TOTAL_GPUS=1

TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/image_list.txt"
find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

# Check if image list was created successfully
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found. Check your input directory and permissions."
  exit 1
fi

# # Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
BASE_IMAGES_PER_JOB=$((NUM_IMAGES / TOTAL_JOBS))
REMAINDER=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Variable to keep track of the current line in the image list
CURRENT_LINE=1

# Divide image paths into text files for each job
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"

  # Determine the number of images for this job
  if [ $i -lt $REMAINDER ]; then
    IMAGES_THIS_JOB=$((BASE_IMAGES_PER_JOB + 1))
  else
    IMAGES_THIS_JOB=$BASE_IMAGES_PER_JOB
  fi

  # Extract the appropriate number of lines for this job
  sed -n "${CURRENT_LINE},$((CURRENT_LINE + IMAGES_THIS_JOB - 1))p" "${IMAGE_LIST}" > "${TEXT_FILE}"

  # Update the current line for the next iteration
  CURRENT_LINE=$((CURRENT_LINE + IMAGES_THIS_JOB))
done

# Run the process on the GPUs, allowing multiple jobs per GPU
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=$((i % TOTAL_GPUS))
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ${RUN_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}" \
    --radius ${RADIUS} \
    --kpt-thr ${KPT_THRES} \
    --thickness ${LINE_THICKNESS} \
    --xy_axis_limit ${XY_AXIS_LIMIT} \
    --z_axis_limit ${Z_AXIS_LIMIT} &
  # Allow a short delay between starting each job to reduce system load spikes
  sleep 1
done

# Wait for all background processes to finish
wait

# Remove the image list text file
rm "${IMAGE_LIST}"

# Go back to the original script's directory
cd -

echo "Processing complete."
# echo "Making a video"

VIDEO_NAME='output'

# Define the functions
make_video() {
    ffmpeg -framerate "$2" -pattern_type glob -i '*.jpg' -pix_fmt yuv420p "$1".mp4
}

make_video_png() {
    ffmpeg -framerate "$2" -pattern_type glob -i '*.png' -pix_fmt yuv420p "$1".mp4
}

# After processing is complete
cd $OUTPUT
rm -rf $VIDEO_NAME.mp4

# Determine the file type and call the appropriate function
FILE_TYPE=$(ls | head -n 1 | awk -F . '{print $NF}')

if [ "$FILE_TYPE" = "jpg" ]; then
    make_video $VIDEO_NAME 30
elif [ "$FILE_TYPE" = "png" ]; then
    make_video_png $VIDEO_NAME 30
else
    echo "Unsupported file type: $FILE_TYPE"
fi

echo "Video ready to copy!"
realpath $VIDEO_NAME.mp4
