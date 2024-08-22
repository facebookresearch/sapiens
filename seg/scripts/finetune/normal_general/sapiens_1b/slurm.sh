#!/bin/bash

####-------------------MODEL_CARD----------------------------
DATASET='normal_general'
MODEL="sapiens_1b_${DATASET}-1024x768"
JOB_NAME="$MODEL"
TRAIN_BATCH_SIZE_PER_GPU=2

RESUME_FROM=''
LOAD_FROM=''

# NUM_NODES=2
NUM_NODES=4

## set you conda environment here.
CONDA_ENV='/home/conda-envs/sapiens/bin/activate'

##------------------------------------------------------------
CONFIG_FILE=configs/sapiens_normal/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/train/${DATASET}/${MODEL}/slurm"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"
TIME='7-00:00:00'
JOB_NAME="${JOB_NAME}"
WORLD_SIZE=$(($NUM_NODES * 8))

##---------------------------------------------------------------
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

###-----------------------------------------------------------------
# Exporting variables
export MODEL NUM_NODES WORLD_SIZE TRAIN_BATCH_SIZE_PER_GPU CONFIG_FILE OPTIONS OUTPUT_DIR
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
srun python tools/train.py \${CONFIG_FILE} --work-dir=\${OUTPUT_DIR} ${CMD_RESUME} --launcher="slurm" --cfg-options \${OPTIONS} | tee \${LOG_FILE}
EOL

###-----------------------------------------------------------------
# Submit the job
sbatch $SCRIPT

# Optionally, remove the temporary script file
rm -f $SCRIPT
