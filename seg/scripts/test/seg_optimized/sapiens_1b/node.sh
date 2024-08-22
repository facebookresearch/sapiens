#!/bin/bash

cd ../../../.. || exit

VALID_GPU_IDS=(0)

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT='/uca/rawalk/sapiens_host/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_epoch_200.pth'
MODEL_NAME='sapiens_1b'; CHECKPOINT='/uca/rawalk/sapiens_host/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth'

OUTPUT='/home/srivathsang/sapiens/seg/Outputs/sapiens_seg_1b_optimized'

DATASET='goliath'
MODEL="${MODEL_NAME}_${DATASET}-1024x768"
CONFIG_FILE="configs/sapiens_seg/${DATASET}/${MODEL}.py"

BATCH_SIZE=32

OPTIONS="$(echo "test_dataloader.batch_size=${BATCH_SIZE}")"

export DB_CACHE_DIR=/shared/airstore_index/avatar_index_cache/

CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} python3 tools/test_optimized.py ${CONFIG_FILE} ${CHECKPOINT} \
                                                                        --work-dir ${OUTPUT} \
                                                                        --cfg-options ${OPTIONS} \
                                                                        --launcher="none"
