cd ../../../..

###--------------------------------------------------------------
# DEVICES=0,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

##---------copy this to the slurm script-----------------
####-----------------MODEL_CARD----------------------------
DATASET='coco_wholebody'
MODEL="sapiens_1b-210e_${DATASET}-1024x768"
JOB_NAME="test_pose_whole_$MODEL"
TEST_BATCH_SIZE_PER_GPU=32

CHECKPOINT="/home/$USER/sapiens_host/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth"

#---------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------+
CONFIG_FILE=configs/sapiens_pose/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/test/${DATASET}/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

export TF_CPP_MIN_LOG_LEVEL=2

## set the options for the test
OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU")"

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
