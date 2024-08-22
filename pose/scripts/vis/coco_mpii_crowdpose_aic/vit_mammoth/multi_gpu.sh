#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='demo/custom_topdown_demo_with_mmdet.py'
DETECTION_CONFIG_FILE='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DETECTION_CHECKPOINT='./checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

DATA_DIR='/home/rawalk/Desktop/foundational/mmpose/data'
OUTPUT_DIR='/home/rawalk/Desktop/foundational/mmpose/Outputs/vis'

####--------------------------vit-mammoth, pretrained shutterstock, trained on goliath + all, 64 nodes---------------
DATASET='coco_mpii_crowdpose_aic'
MODEL="vit_mammoth_8xb64-210e_${DATASET}-1024x768"

CONFIG_FILE="configs/body_2d_keypoint/fm_topdown_heatmap/${DATASET}/${MODEL}.py"
CHECKPOINT='/home/rawalk/drive/mmpose/Outputs/train/coco_mpii_crowdpose_aic/vit_mammoth_8xb64-210e_coco_mpii_crowdpose_aic-1024x768/slurm/11-06-2023_00:21:11/best_coco_AP_epoch_42.pth'

# ###--------------------------------data:goliath--------------------------------------------------
# SESSION_ID='s--20190524--1430--4911137--GHS'
# # CAMERA_ID='400021'
# # CAMERA_ID='400130'
# # CAMERA_ID='400156'
# CAMERA_ID='400190'

# DATA_PREFIX="goliath/test/images/${SESSION_ID}/${CAMERA_ID}"
# LINE_THICKNESS=5 ## 1 is default
# RADIUS=4

# # ###--------------------------------data:goliath--------------------------------------------------
# DATA_PREFIX="goliath/full_body/subject1/cam400167"
# LINE_THICKNESS=5 ## 1 is default
# RADIUS=4

# ###--------------------------------data:hand--------------------------------------------------
# DATA_PREFIX="goliath/hands/images"
# LINE_THICKNESS=2
# RADIUS=1

# # # # ##------------------------------data:in the wild---------------------------------------
DATA_PREFIX="in_the_wild/video1/images"
LINE_THICKNESS=3
RADIUS=2

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

echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} text files."

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
    ${DETECTION_CONFIG_FILE} \
    ${DETECTION_CHECKPOINT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}" \
    --radius ${RADIUS} \
    --thickness ${LINE_THICKNESS} &
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
