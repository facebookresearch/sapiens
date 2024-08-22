cd ../../../..

###--------------------------------------------------------------
# DEVICES=0,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

##---------copy this to the slurm script-----------------
####-----------------MODEL_CARD----------------------------
DATASET='coco_wholebody'
MODEL="vit_mammoth_8xb64-210e_${DATASET}-1024x768"
JOB_NAME='pose_whole_td-hm_hrnet-w48_dark_384x288'
TEST_BATCH_SIZE_PER_GPU=64

CHECKPOINT='/home/rawalk/drive/mmpose/Outputs/train/coco_wholebody/vit_mammoth_8xb64-210e_coco_wholebody-1024x768/slurm/11-13-2023_16:55:40/best_coco-wholebody_AP_epoch_210.pth'

##--------------------------------------------------------------
# ## val set
# EVAL_SET='val2017'; DETECTION_AP=56 ## default for val
EVAL_SET='val2017'; DETECTION_AP=70 ## best for val

## test-dev set
# EVAL_SET='test-dev2017'; DETECTION_AP=609 ## default for test set, 378148 bboxes
# EVAL_SET='test-dev2017'; DETECTION_AP=0 ## best for test set by eva02

#---------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------+
BBOX_FILE=data/coco/person_detection_results/COCO_${EVAL_SET}_detections_AP_H_${DETECTION_AP}_person.json
CONFIG_FILE=configs/body_2d_keypoint/fm_topdown_heatmap/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/test/${DATASET}/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

###--------------------------------------------------------------
if [ "$EVAL_SET" = "val2017" ]; then
    ANNOTATION_FILE=annotations/coco_wholebody_val_v1.0.json
    IMG_PREFIX=${EVAL_SET}/

else
    ANNOTATION_FILE=annotations/image_info_test-dev2017.json
    IMG_PREFIX=test2017/
fi

export TF_CPP_MIN_LOG_LEVEL=2

## set the options for the test
OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU
    test_dataloader.dataset.bbox_file=${BBOX_FILE}
    test_dataloader.dataset.ann_file=$ANNOTATION_FILE
    test_dataloader.dataset.data_prefix.img=${IMG_PREFIX}
    test_evaluator.ann_file=data/coco/$ANNOTATION_FILE
    ")"

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TEST_BATCH_SIZE_PER_GPU=16 ## works for single gpu

    OPTIONS="$(echo "test_dataloader.batch_size=${TEST_BATCH_SIZE_PER_GPU} test_dataloader.num_workers=0 test_dataloader.persistent_workers=False")"
    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} ${CHECKPOINT}\
            ${NUM_GPUS} \
            --work-dir ${OUTPUT_DIR} \
            --cfg-options ${OPTIONS} \
            | tee ${LOG_FILE}

fi
