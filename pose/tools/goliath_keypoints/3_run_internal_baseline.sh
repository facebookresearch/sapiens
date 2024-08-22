#!/bin/bash
cd ../../../internal_baselines/keypoint_detector

# Define base directories
INPUT_BASE_DIR="/mnt/home/rawalk/Desktop/foundational/mmpose/data/goliath/test/chunks"
OUTPUT_BASE_DIR="/mnt/home/rawalk/drive/mmpose/Outputs/goliath/test_chunks"
MODEL_PATH="/mnt/home/hewen/codes/genesis/general_keypoint_detector/runs/ala_344-8x-enb0-256x3-2k-rsc/final_model.pth"

# Loop over chunk IDs
for CHUNK_ID in {0..10}
do
    # Define input and output paths for the current chunk
    INPUT_LIST="${INPUT_BASE_DIR}/images_${CHUNK_ID}.txt"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHUNK_ID}"

    # Echo current input and output
    echo "Processing chunk ${CHUNK_ID}"
    echo "Input list: $INPUT_LIST"
    echo "Output dir: $OUTPUT_DIR"

    # # Run the prediction script
    ./predict.py --model $MODEL_PATH \
    --list ${INPUT_LIST} \
    --output-dir ${OUTPUT_DIR} \
    -b24 -j8 --gpu 1,2,3,4,5,6 \
    --verbose \
    --output-type 'pth' \
    --combine-results

    # Check if the prediction script exited successfully
    if [ $? -ne 0 ]; then
        echo "Error processing chunk ${CHUNK_ID}. Exiting."
        exit 1
    fi

    echo "Completed chunk ${CHUNK_ID}"
done

echo "All chunks processed."
