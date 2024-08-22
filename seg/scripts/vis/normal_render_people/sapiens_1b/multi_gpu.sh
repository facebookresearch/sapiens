#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/demo_normal_vis.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis'

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DATASET='normal_render_people'
MODEL="sapiens_1b_${DATASET}-1024x768"

CONFIG_FILE="configs/sapiens_normal/${DATASET}/${MODEL}.py"
CHECKPOINT='/uca/rawalk/sapiens_host/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115.pth'

# ##------------------------------data:nvidia in the wild---------------------------------------
# DATA_PREFIX="nvidia_in_the_wild/69000_small"
# SEG_DIR='/home/rawalk/Desktop/foundational/mmseg/Outputs/vis/goliath/vit_mammoth_8xb1-160k_goliath-1024x1024/nvidia_in_the_wild/69000_tiny/epoch_150_output'

# DATA_PREFIX="nvidia_in_the_wild/69000"
# SEG_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapien_1b_goliath-1024x768/nvidia_in_the_wild/69000/sapien_1b_goliath_epoch_150_output'

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
DATA_PREFIX="itw_videos/reel11"
# DATA_PREFIX="itw_videos/reel22"
# DATA_PREFIX="itw_videos/reel40"
# DATA_PREFIX="itw_videos/reel36"
# DATA_PREFIX="itw_videos/reel37"
# # DATA_PREFIX="itw_videos/reel35"
# DATA_PREFIX="itw_videos/reel38"

# SEG_DIR=/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/$DATA_PREFIX/sapiens_1b_goliath_epoch_150_output
SEG_DIR=/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/$DATA_PREFIX/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_output

# ##------------------------------------------------------------------------------
# DATA_DIR='/home/rawalk/Desktop/sapiens/pretrain/data'
# DATA_PREFIX="shutterstock"
# SEG_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/shutterstock/sapiens_1b_goliath_epoch_150_output'

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

##-----------------------------------------------------------------------
echo "Making a video"

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
