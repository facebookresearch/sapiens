#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/demo_normal_vis.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis2'

####-----------------------------------------
DATASET='normal_render_people'
MODEL="sapiens2_0.6b_${DATASET}-4096x3072"

CONFIG_FILE="configs/sapiens2_normal/${DATASET}/${MODEL}.py"
# CHECKPOINT='/home/rawalk/drive/seg/Outputs/train2/normal_render_people/sapiens2_0.6b_normal_render_people-4096x3072/slurm/07-25-2024_04:30:41/iter_16000.pth'
CHECKPOINT='/home/rawalk/drive/seg/Outputs/train2/normal_render_people/sapiens2_0.6b_normal_render_people-4096x3072/slurm/07-25-2024_04:30:41/iter_25000.pth'

# ###--------------------------------data:goliath--------------------------------------------------
# DATA_DIR='/home/rawalk/Desktop/foundational/mmpose/data'
# DATA_PREFIX="goliath/full_body/subject1/cam400167"

###--------------------------------data:hand--------------------------------------------------
# INPUT='tests/data/internal_hand'
# OUTPUT_ROOT='/home/rawalk/Desktop/foundational/mmpose/Outputs/vis/eva02_large/internal_hand'

# ##------------------------------data:nvidia in the wild---------------------------------------
DATA_PREFIX="nvidia_in_the_wild/69000_small"
SEG_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/nvidia_in_the_wild/69000/sapiens_1b_goliath_epoch_150_output'

# DATA_PREFIX="nvidia_in_the_wild/69000"
# SEG_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/nvidia_in_the_wild/69000/sapiens_1b_goliath_epoch_150_output'

# # ##------------------------------data:itw videos---------------------------------------
# DATA_PREFIX="itw_videos/reel1"
# DATA_PREFIX="itw_videos/reel2"
# DATA_PREFIX="itw_videos/reel3"
# DATA_PREFIX="itw_videos/reel4"
# DATA_PREFIX="itw_videos/reel5"
# DATA_PREFIX="itw_videos/reel6"
# DATA_PREFIX="itw_videos/reel7"
# # DATA_PREFIX="itw_videos/reel8"
# DATA_PREFIX="itw_videos/reel9"
# DATA_PREFIX="itw_videos/reel10"
# DATA_PREFIX="itw_videos/reel11"

# SEG_DIR=/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapien_1b_goliath-1024x768/$DATA_PREFIX/sapien_1b_goliath_epoch_150_output

##------------------------------------------------------------------------------
INPUT="${DATA_DIR}/${DATA_PREFIX}"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}" .pth)
OUTPUT="${OUTPUT_DIR}/${DATASET}/${MODEL}/${DATA_PREFIX}/${CHECKPOINT_NAME}_output"

JOBS_PER_GPU=4
TOTAL_GPUS=8
TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/image_list.txt"
find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

# Check if image list was created successfully
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found. Check your input directory and permissions."
  exit 1
fi

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Divide image paths into text files for each job
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
  if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
    # For the last text file, write all remaining image paths
    tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
  else
    # Write the exact number of image paths per text file
    head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
  fi
done

# Run the process on the GPUs, allowing multiple jobs per GPU
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=$((i % TOTAL_GPUS))
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ${RUN_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --seg_dir ${SEG_DIR} \
    --output-root="${OUTPUT}" &
  # Allow a short delay between starting each job to reduce system load spikes
  sleep 1
done

# Wait for all background processes to finish
wait

# Remove the image list and temporary text files
rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${INPUT}/image_paths_$((i+1)).txt"
done

# Go back to the original script's directory
cd -

echo "Processing complete."
