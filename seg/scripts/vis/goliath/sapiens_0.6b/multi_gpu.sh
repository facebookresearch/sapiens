#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/custom_image_save_mask_demo.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis'

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DATASET='goliath'
MODEL="sapien_0.6b_${DATASET}-1024x768"

CONFIG_FILE="configs/sapiens_seg/${DATASET}/${MODEL}.py"
CHECKPOINT='/uca/rawalk/sapiens/seg/checkpoints/sapien_0.6b/sapien_0.6b_goliath_epoch_200.pth'

# ###--------------------------------data:goliath--------------------------------------------------
# DATA_DIR='/home/rawalk/Desktop/foundational/mmpose/data'
# DATA_PREFIX="goliath/full_body/subject1/cam400167"

###--------------------------------data:hand--------------------------------------------------
# INPUT='tests/data/internal_hand'
# OUTPUT_ROOT='/home/rawalk/Desktop/foundational/mmpose/Outputs/vis/eva02_large/internal_hand'

# # # ##------------------------------data:in the wild---------------------------------------
# DATA_DIR='/home/rawalk/Desktop/foundational/mmpose/data'
# DATA_PREFIX="in_the_wild/video1/images"

# ##------------------------------data:cihp---------------------------------------
# DATA_PREFIX="cihp/Testing/Images_small"

# ##------------------------------data:lip---------------------------------------
# DATA_PREFIX="lip/images/testing_images_small"

# # # ##------------------------------data:nvidia in the wild---------------------------------------
DATA_PREFIX="nvidia_in_the_wild/69000_small"
# DATA_PREFIX="nvidia_in_the_wild/69000_tiny"
# DATA_PREFIX="nvidia_in_the_wild/69000"

##---------------egohumans-----------------
# DATA_PREFIX="egohumans/rgb"
# DATA_PREFIX="egohumans/images/rgb"
# DATA_PREFIX="egohumans/images/left"
# DATA_PREFIX="egohumans/ego/aria01/images/rgb"

# # ##------------------------------data:itw videos---------------------------------------
# DATA_PREFIX="itw_videos/reel1"
# DATA_PREFIX="itw_videos/reel2"
# DATA_PREFIX="itw_videos/reel3"
# DATA_PREFIX="itw_videos/reel4"
# DATA_PREFIX="itw_videos/reel5"
# DATA_PREFIX="itw_videos/reel6"
# DATA_PREFIX="itw_videos/reel7"
# DATA_PREFIX="itw_videos/reel8"
# DATA_PREFIX="itw_videos/reel9"
# DATA_PREFIX="itw_videos/reel10"
# DATA_PREFIX="itw_videos/reel11"

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

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
echo "Output directory: ${OUTPUT}"
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
