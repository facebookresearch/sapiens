#!/bin/bash

cd ../../../../pose || exit

#----------------------------set your input and output directories----------------------------------------------
# Set the default value for ICA_DIR
ICA_DIR='/uca/full_head_ICA/2022-10-19/Chen/Segs/BodySpin/image'
JOBS_PER_GPU=4; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)

# If an argument is provided, use it as ICA_DIR
if [ $# -eq 3 ]; then
    ICA_DIR="$1"
    JOBS_PER_GPU=$2
    VALID_GPU_IDS=($3)  # Convert the third argument into an array
fi

TASK='pose'

INPUT=$ICA_DIR/iPhone_rgb
OUTPUT=$ICA_DIR/$TASK

#--------------------------MODEL CARD---------------
MODEL_NAME='sapiens_1b'; CHECKPOINT='/uca/rawalk/sapiens_host/pose/checkpoints/sapiens_1b/sapiens_1b_goliath_coco_wholebody_mpii_crowdpose_aic_best_goliath_AP_630.pth'

DATASET='goliath_coco_wholebody_mpii_crowdpose_aic'
MODEL="${MODEL_NAME}-210e_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_pose/${DATASET}/${MODEL}.py"

# bounding box detector
DETECTION_CONFIG_FILE='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DETECTION_CHECKPOINT='/uca/rawalk/sapiens_host/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

#---------------------------VISUALIZATION PARAMS--------------------------------------------------
LINE_THICKNESS=10 ## line thickness of the skeleton
RADIUS=10 ## keypoint radius

KPT_THRES=0.3 ## default keypoint confidence
# KPT_THRES=0.5 ## higher keypoint confidence

##-------------------------------------inference-------------------------------------
RUN_FILE='demo/demo_vis.py'

## number of inference jobs per gpu, total number of gpus and gpu ids
TOTAL_GPUS=${#VALID_GPU_IDS[@]}
TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/${TASK}_image_list.txt"
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
  TEXT_FILE="${INPUT}/${TASK}_image_paths_$((i+1)).txt"
  START_LINE=$((IMAGES_PER_FILE * i + (i < EXTRA_IMAGES ? i : EXTRA_IMAGES) + 1))
  END_LINE=$((START_LINE + IMAGES_PER_FILE + (i < EXTRA_IMAGES ? 1 : 0) - 1))
  sed -n "${START_LINE},${END_LINE}p" "${IMAGE_LIST}" > "${TEXT_FILE}"
done

# Run the process on the GPUs, allowing multiple jobs per GPU
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=$((i % TOTAL_GPUS))
  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} python ${RUN_FILE} \
    ${DETECTION_CONFIG_FILE} \
    ${DETECTION_CHECKPOINT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input "${INPUT}/${TASK}_image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}" \
    --save-predictions \
    --radius ${RADIUS} \
    --kpt-thr ${KPT_THRES} \
    --thickness ${LINE_THICKNESS} &
  # Allow a short delay between starting each job to reduce system load spikes
  sleep 1
done

# Wait for all background processes to finish
wait

# Remove the image list and temporary text files
rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${INPUT}/${TASK}_image_paths_$((i+1)).txt"
done
# Go back to the original script's directory
cd -

echo "Processing complete."
echo "Making a video"

VIDEO_NAME=$TASK

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

mkdir -p $ICA_DIR/videos
mv $VIDEO_NAME.mp4 $ICA_DIR/videos
cd $ICA_DIR/videos

echo "Video ready to copy!"
realpath $VIDEO_NAME.mp4
