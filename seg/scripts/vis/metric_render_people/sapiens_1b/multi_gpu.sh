#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/demo_metric_depth_vis.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis'

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, iter 600, 64 nodes---------------
DATASET='metric_render_people'
MODEL="sapiens_1b_${DATASET}-1024x768"
SEG_DIR=''

CONFIG_FILE="configs/sapiens_depth/${DATASET}/${MODEL}.py"
# CHECKPOINT='/home/rawalk/drive/seg/Outputs/train/metric_render_people/sapiens_1b_metric_render_people-1024x768/slurm/04-27-2024_01:47:29/epoch_6.pth'
# CHECKPOINT='/home/rawalk/drive/seg/Outputs/train/metric_render_people/sapiens_1b_metric_render_people-1024x768/slurm/04-20-2024_04:51:04/epoch_90.pth'
CHECKPOINT='/home/rawalk/drive/seg/Outputs/train/metric_render_people/sapiens_1b_metric_render_people-1024x768/slurm/04-20-2024_04:51:04/epoch_50.pth'

# ##------------------------------data:nvidia in the wild---------------------------------------
DATA_PREFIX="nvidia_in_the_wild/69000_small"
# # DATA_PREFIX="nvidia_in_the_wild/69000"
SEG_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/nvidia_in_the_wild/69000/sapiens_1b_goliath_epoch_150_output'

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
# DATA_PREFIX="itw_videos/reel22"
# DATA_PREFIX="itw_videos/reel40"
# DATA_PREFIX="itw_videos/reel36"
# DATA_PREFIX="itw_videos/reel37"
# DATA_PREFIX="itw_videos/reel35"
# DATA_PREFIX="itw_videos/reel38"
# DATA_PREFIX="itw_videos/reel17"
# DATA_PREFIX="itw_videos/reel24"
# DATA_PREFIX="itw_videos/reel26"

## face
# DATA_PREFIX="itw_videos/reel38"
# DATA_PREFIX="itw_videos/reel39"
# # DATA_PREFIX="itw_videos/reel10"
# DATA_PREFIX="itw_videos/reel17"

# SEG_DIR=/home/rawalk/Desktop/sapiens/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/$DATA_PREFIX/sapiens_1b_goliath_epoch_150_output

# ##--------------------render_people------------------
# DATA_PREFIX="render_people/synthetic/rgb_small"
# SEG_DIR='/home/rawalk/drive/seg/Outputs/vis/goliath/sapiens_1b_goliath-1024x768/render_people/synthetic/rgb_small/sapiens_1b_goliath_epoch_150_output'

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

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Divide image paths into text files for each job
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
  START_LINE=$((IMAGES_PER_FILE * i + (i < EXTRA_IMAGES ? i : EXTRA_IMAGES) + 1))
  END_LINE=$((START_LINE + IMAGES_PER_FILE + (i < EXTRA_IMAGES ? 1 : 0) - 1))
  sed -n "${START_LINE},${END_LINE}p" "${IMAGE_LIST}" > "${TEXT_FILE}"
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

# Determine the file type, excluding .npy files
FILE_TYPE=$(ls | grep -E '\.jpg$|\.png$' | head -n 1 | awk -F . '{print $NF}')

if [ "$FILE_TYPE" = "jpg" ]; then
    # make_video $VIDEO_NAME 1
    make_video $VIDEO_NAME 30
elif [ "$FILE_TYPE" = "png" ]; then
    # make_video_png $VIDEO_NAME 1
    make_video_png $VIDEO_NAME 30
else
    echo "Unsupported file type: $FILE_TYPE"
fi

echo "Video ready to copy!"
realpath $VIDEO_NAME.mp4
