#!/bin/bash

####-------------------MODEL_CARD----------------------------
DATASET='coco_wholebody'
MODEL="sapien_2b-210e_${DATASET}-1024x768"
JOB_NAME="test_pose_whole_$MODEL"
TEST_BATCH_SIZE_PER_GPU=32

CHECKPOINT='/uca/rawalk/sapiens/pose/checkpoints/sapien_2b/sapien_2b_coco_wholebody_best_coco_wholebody_AP_745.pth'

EVAL_SET='val2017'; DETECTION_AP=70 ## best for val
# EVAL_SET='val2017'; DETECTION_AP=56 ## default

# NUM_NODES=8
# NUM_NODES=16
# NUM_NODES=32
NUM_NODES=64

##------------------------------------------------------------
CONFIG_FILE=configs/sapiens_pose/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/test/${DATASET}/${MODEL}/slurm"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"
BBOX_FILE=data/coco/person_detection_results/COCO_${EVAL_SET}_detections_AP_H_${DETECTION_AP}_person.json
CONDA_ENV='/uca/conda-envs/dgxenv-2023-09-25-7853/bin/activate'
TIME='7-00:00:00'
JOB_NAME="${JOB_NAME}"
WORLD_SIZE=$(($NUM_NODES * 8))

###--------------------------------------------------------------
if [ "$EVAL_SET" = "val2017" ]; then
    ANNOTATION_FILE=annotations/coco_wholebody_val_v1.0.json
    IMG_PREFIX=${EVAL_SET}/

else
    ANNOTATION_FILE=annotations/image_info_test-dev2017.json
    IMG_PREFIX=test2017/
fi

OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU
    test_dataloader.dataset.bbox_file=${BBOX_FILE}
    test_dataloader.dataset.ann_file=$ANNOTATION_FILE
    test_dataloader.dataset.data_prefix.img=${IMG_PREFIX}
    test_evaluator.ann_file=data/coco/$ANNOTATION_FILE
    ")"

###-----------------------------------------------------------------
# Exporting variables
export MODEL NUM_NODES WORLD_SIZE TRAIN_BATCH_SIZE_PER_GPU CONFIG_FILE OPTIONS OUTPUT_DIR CHECKPOINT
export TF_CPP_MIN_LOG_LEVEL=2

ROOT_DIR=$(cd ../../../.. && pwd)

###-----------------------------------------------------------------
# Create a temporary script file
SCRIPT=$(mktemp)

# Write the SLURM script to the temporary file
cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=learn
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --output=${ROOT_DIR}/${OUTPUT_DIR}/slurm/%j.out
#SBATCH --error=${ROOT_DIR}/${OUTPUT_DIR}/slurm/%j.err
#SBATCH --exclude=avalearn1992  # Exclude the node avalearn1992

cd $ROOT_DIR

source ${CONDA_ENV}

LOG_FILE="\${OUTPUT_DIR}/log.txt"
mkdir -p \${OUTPUT_DIR}/slurm
touch \${LOG_FILE}

# Set environment variables for distributed training
export MASTER_PORT=\$(($RANDOM % 31337 + 10000))
master_addr=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=\$master_addr

echo "NODELIST=\${SLURM_NODELIST}"
echo "MASTER_ADDR=\${MASTER_ADDR}"
echo "MASTER_PORT=\${MASTER_PORT}"
echo "WORLD_SIZE=\${WORLD_SIZE}"
echo "CUDA_HOME=\$CUDA_HOME"

PYTHONPATH="\$(dirname \$0)/..":\$PYTHONPATH \
srun python tools/test.py \${CONFIG_FILE} \${CHECKPOINT} --work-dir=\${OUTPUT_DIR} --launcher="slurm" --cfg-options \${OPTIONS} | tee \${LOG_FILE}
EOL

###-----------------------------------------------------------------
# Submit the job
sbatch $SCRIPT

# Optionally, remove the temporary script file
rm -f $SCRIPT
