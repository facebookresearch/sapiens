#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=/home/${USER}/sapiens_lite_host

MODE='torchscript' ## original. no optimizations (slow). full precision inference.
# MODE='bfloat16' ## A100 gpus. faster inference at bfloat16

SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

#----------------------------set your input and output directories----------------------------------------------
INPUT='../pose/demo/data/itw_videos/video.mp4'
OUTPUT="/home/${USER}/Desktop/sapiens/pose/Outputs/vis/itw_videos/video"

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796_$MODE.pt2
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_best_coco_AP_812_$MODE.pt2
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_$MODE.pt2
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_best_coco_AP_822_$MODE.pt2

OUTPUT=$OUTPUT/$MODEL_NAME

DETECTION_CONFIG_FILE='../pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

#---------------------------VISUALIZATION PARAMS--------------------------------------------------
LINE_THICKNESS=3 ## line thickness of the skeleton
RADIUS=6 ## keypoint radius
KPT_THRES=0.3 ## confidence threshold

##-------------------------------------inference-------------------------------------
RUN_FILE='demo/vis_pose.py'

## number of inference jobs per gpu, total number of gpus and gpu ids
JOBS_PER_GPU=1; TOTAL_GPUS=1; VALID_GPU_ID=0
BATCH_SIZE=8

# Check if video exists
if [ ! -s "${INPUT}" ]; then
  echo "No video found. Check your input directory and permissions."
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=2

GPU_ID=$((i % TOTAL_GPUS))
CUDA_VISIBLE_DEVICES=${VALID_GPU_ID} python ${RUN_FILE} \
  ${CHECKPOINT} \
  --num_keypoints 17 \
  --det-config ${DETECTION_CONFIG_FILE} \
  --det-checkpoint ${DETECTION_CHECKPOINT} \
  --batch-size ${BATCH_SIZE} \
  --input "${INPUT}" \
  --output-root="${OUTPUT}" \
  --radius ${RADIUS} \
  --kpt-thr ${KPT_THRES} ## add & to process in background


# Go back to the original script's directory
cd -

echo "Processing complete."
echo "Results saved to $OUTPUT"
