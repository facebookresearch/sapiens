#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/demo_seg_vis.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis'

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DATASET='goliath'
MODEL="sapiens_1b_${DATASET}-1024x768"

CONFIG_FILE="configs/sapiens_seg/${DATASET}/${MODEL}.py"
CHECKPOINT='/uca/rawalk/sapiens_host/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth'


# # # ##------------------------------data:nvidia in the wild---------------------------------------
# DATA_PREFIX="nvidia_in_the_wild/69000_tiny"
# DATA_PREFIX="nvidia_in_the_wild/69000_small"
# DATA_PREFIX="nvidia_in_the_wild/69000"

##---------------egohumans-----------------
# DATA_PREFIX="egohumans/rgb"
# DATA_PREFIX="egohumans/images/rgb"
# DATA_PREFIX="egohumans/images/left"
# DATA_PREFIX="egohumans/ego/aria01/images/rgb"

# # ##------------------------------data:itw videos---------------------------------------
# DATA_PREFIX="itw_videos/reel01"
# DATA_PREFIX="itw_videos/reel02"
# DATA_PREFIX="itw_videos/reel03"
# DATA_PREFIX="itw_videos/reel04"
DATA_PREFIX="itw_videos/reel11"

##----------------render_people----------------------
# DATA_PREFIX="render_people/synthetic/test/stereo_rgb_small/rp_kumar_posed_001_100k"

##---------------shutterstock----------------
# DATA_DIR='/home/rawalk/Desktop/sapiens/pretrain/data'
# DATA_PREFIX="shutterstock"

# ##-------------itw_turntable--------------
# # DATA_PREFIX='itw_turntable/01'
# # DATA_PREFIX='itw_turntable/02'
# DATA_PREFIX='itw_turntable/03'

##------------------------------------------------------------------------------
INPUT="${DATA_DIR}/${DATA_PREFIX}"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}" .pth)
OUTPUT="${OUTPUT_DIR}/${DATASET}/${MODEL}/${DATA_PREFIX}/${CHECKPOINT_NAME}_output"

JOBS_PER_GPU=4
TOTAL_GPUS=8
TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/image_list.txt"
find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png -o -iname \*.jpeg \) | sort > "${IMAGE_LIST}"

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
FILE_TYPE=$(ls | grep -E '\.jpg$|\.png$' | head -n 1 | awk -F . '{print $NF}')

if [ "$FILE_TYPE" = "jpg" ]; then
    make_video $VIDEO_NAME 30
elif [ "$FILE_TYPE" = "png" ]; then
    make_video_png $VIDEO_NAME 30
else
    echo "Unsupported file type: $FILE_TYPE"
fi

echo "Video ready to copy!"
realpath $VIDEO_NAME.mp4
