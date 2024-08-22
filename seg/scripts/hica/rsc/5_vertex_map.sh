#!/bin/bash

cd ../../.. || exit

#----------------------------set your input and output directories----------------------------------------------
TASK='vertex_map'
INPUT='demo/data/mgr'
OUTPUT=/home/rawalk/Desktop/sapiens/seg/Outputs/vis/hica/$TASK

# JOBS_PER_GPU=4; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
JOBS_PER_GPU=1; VALID_GPU_IDS=(0)

#--------------------------MODEL CARD---------------
MODEL_NAME='sapiens_0.6b';
CHECKPOINT=/uca/rawalk/sapiens_hica/$TASK/checkpoints/$MODEL_NAME/$MODEL_NAME'_'$TASK'_'epoch_3.pth

DATASET=$TASK
MODEL="${MODEL_NAME}_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_hica/${DATASET}/${MODEL}.py"

##-------------------------------------inference-------------------------------------
RUN_FILE='demo/hica/5_demo_vertex_map_vis.py'

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
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input "${INPUT}/${TASK}_image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}"
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