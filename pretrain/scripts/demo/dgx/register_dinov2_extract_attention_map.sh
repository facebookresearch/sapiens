#!/bin/bash

cd ../../../.. || exit

#----------------------------set your input and output directories----------------------------------------------
# IMAGE_DIR='attention1'
# IMAGE_DIR='attention3'
IMAGE_DIR='attention4'

INPUT=./demo/data/$IMAGE_DIR
OUTPUT=/mnt/home/rawalk/Desktop/sapiens/pretrain/Outputs/vis/registers_dinov2_$IMAGE_DIR

#--------------------------MODEL CARD---------------
DATASET='shutterstock_instagram_test'
MODEL="mae_sapiens_1b-p16_8xb512-coslr-1600e_${DATASET}"

CONFIG_FILE="configs/sapiens_mae/${DATASET}/${MODEL}.py"
CHECKPOINT='/uca/rawalk/sapiens/pretrain/checkpoints/sapiens_1b/sapiens_1b_shutterstock_instagram_epoch_173.pth'

##-------------------------------------inference-------------------------------------
RUN_FILE='demo/registers_dinov2_extract_attention_map.py'

# JOBS_PER_GPU=4 ## number of inference jobs per gpu
# TOTAL_GPUS=8 ## total number of gpus

JOBS_PER_GPU=1 ## number of inference jobs per gpu
TOTAL_GPUS=1 ## total number of gpus

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
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}"
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
echo "Results saved to $OUTPUT"
