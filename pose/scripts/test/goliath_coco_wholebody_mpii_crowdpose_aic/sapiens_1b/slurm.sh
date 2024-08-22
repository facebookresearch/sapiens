#!/bin/bash

####-------------------MODEL_CARD----------------------------
DATASET='goliath_coco_wholebody_mpii_crowdpose_aic'
MODEL="sapiens_1b-210e_${DATASET}-1024x768"
JOB_NAME="eval_pose_$MODEL"
TEST_BATCH_SIZE_PER_GPU=32

CHECKPOINT='/home/rawalk/drive/pose/Outputs/train/goliath_coco_wholebody_mpii_crowdpose_aic/sapiens_1b-210e_goliath_coco_wholebody_mpii_crowdpose_aic-1024x768/slurm/03-15-2024_22:56:16/best_goliath_AP_iter_22596.pth'

# NUM_NODES=8
NUM_NODES=16
# NUM_NODES=32
# NUM_NODES=64

##------------------------------------------------------------
CONFIG_FILE=configs/sapiens_pose/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/test/${DATASET}/${MODEL}/slurm"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"
CONDA_ENV='/uca/conda-envs/dgxenv-2023-09-25-7853/bin/activate'
TIME='7-00:00:00'
JOB_NAME="${JOB_NAME}"
WORLD_SIZE=$(($NUM_NODES * 8))

###--------------------------------------------------------------
OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU")"

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
