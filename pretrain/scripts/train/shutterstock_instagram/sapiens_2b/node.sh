cd ../../../..

###--------------------------------------------------------------
# DEVICES=0,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# ###----------------------------------------------------
DATASET='shutterstock_instagram'
MODEL="mae_sapiens_2b-p16_8xb512-coslr-1600e_${DATASET}"
TRAIN_BATCH_SIZE_PER_GPU=5

##--------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------
CONFIG_FILE=configs/sapiens_mae/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/train/${DATASET}/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

##------------------------------------------------------------------
RESUME_FROM='' ## default, if given '' it is ignored

LOAD_FROM=''

###--------------------------------------------------------------
if [ -n "$LOAD_FROM" ]; then
    OPTIONS="train_dataloader.batch_size=$TRAIN_BATCH_SIZE_PER_GPU load_from=$LOAD_FROM"
else
    OPTIONS="train_dataloader.batch_size=$TRAIN_BATCH_SIZE_PER_GPU"
fi

if [ -n "$RESUME_FROM" ]; then
    CMD_RESUME="--resume ${RESUME_FROM}"
else
    CMD_RESUME=""
fi

##--------------------------------------------------------------
LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

export TF_CPP_MIN_LOG_LEVEL=2

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    DEBUG_TRAIN_BATCH_SIZE_PER_GPU=8 ## works for single gpu
    OPTIONS="${OPTIONS/train_dataloader.batch_size=$TRAIN_BATCH_SIZE_PER_GPU/train_dataloader.batch_size=$DEBUG_TRAIN_BATCH_SIZE_PER_GPU}"
    OPTIONS="${OPTIONS} train_dataloader.num_workers=0 train_dataloader.persistent_workers=False" # Add this line to append the option for num_workers

    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} --work-dir ${OUTPUT_DIR} ${CMD_RESUME} --no-validate --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
            ${NUM_GPUS} \
            --work-dir ${OUTPUT_DIR} \
            ${CMD_RESUME} \
            --cfg-options ${OPTIONS} \
            | tee ${LOG_FILE}

fi
