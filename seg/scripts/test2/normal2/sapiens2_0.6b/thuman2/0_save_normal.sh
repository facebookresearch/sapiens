#!/bin/bash

# Navigate two levels up from the current directory
cd ../../../../.. || exit

# Set the configurations for detection and pose estimation
RUN_FILE='tools/normal/save_normal.py'

DATA_DIR='/home/rawalk/Desktop/sapiens/seg/data'
OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/test'

####-----------------------------------------
DATASET='normal_render_people'
MODEL="sapiens2_0.6b_${DATASET}-4096x3072"

CONFIG_FILE="configs/sapiens2_normal/${DATASET}/${MODEL}.py"

# CHECKPOINT=/home/rawalk/drive/seg/Outputs/train2/normal_render_people/sapiens2_0.6b_normal_render_people-4096x3072/slurm/07-25-2024_04:30:41/iter_24000.pth
# CHECKPOINT=/home/rawalk/drive/seg/Outputs/train2/normal_render_people/sapiens2_0.6b_normal_render_people-4096x3072/slurm/07-25-2024_04:30:41/iter_16000.pth
CHECKPOINT=/home/rawalk/drive/seg/Outputs/train2/normal_render_people/sapiens2_0.6b_normal_render_people-4096x3072/slurm/07-25-2024_04:30:41/iter_17000.pth

###---------------------------------------------------------------------------------
# Array of dataset parts
DATASET_PARTS=("face" "full_body" "upper_half")

# Define the number of jobs per GPU and total GPUs
JOBS_PER_GPU=4
TOTAL_GPUS=8
VALID_GPU_IDS=(0 1 2 3 4 5 6 7)

for PART in "${DATASET_PARTS[@]}"; do
  DATA_PREFIX="thuman2/evaluation/${PART}/rgb"
  SEG_DIR="${DATA_DIR}/thuman2/evaluation/${PART}/mask"

  INPUT="${DATA_DIR}/${DATA_PREFIX}"
  CHECKPOINT_NAME=$(basename "${CHECKPOINT}" .pth)
  OUTPUT="${OUTPUT_DIR}/${DATASET}/${MODEL}/${DATA_PREFIX}"

  TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))

  # Find all images and sort them, then write to a temporary text file
  IMAGE_LIST="${INPUT}/image_list.txt"
  find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

  # Check if image list was created successfully
  if [ ! -s "${IMAGE_LIST}" ]; then
    echo "No images found in ${INPUT}. Check your input directory and permissions."
    exit 1
  fi

  # Count images and calculate the number of images per text file
  NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
  IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
  EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

  export TF_CPP_MIN_LOG_LEVEL=2
  echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs for ${PART}."

  # Divide image paths into text files for each job
  for ((i=0; i<TOTAL_JOBS; i++)); do
    TEXT_FILE="${INPUT}/${PART}_image_paths_$((i+1)).txt"
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
    GPU_INDEX=$((i % TOTAL_GPUS))
    CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_INDEX]} python ${RUN_FILE} \
      ${CONFIG_FILE} \
      ${CHECKPOINT} \
      --input "${INPUT}/${PART}_image_paths_$((i+1)).txt" \
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
    rm "${INPUT}/${PART}_image_paths_$((i+1)).txt"
  done

  echo "Processing complete for ${PART}."
done

# Go back to the original script's directory
cd -

echo "All processing complete."
