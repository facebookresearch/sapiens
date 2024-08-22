#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables
DATASET='imagenet'
MODEL="mae_sapiens_0.3b-p16_8xb512-coslr-1600e_${DATASET}"

TRAIN_BATCH_SIZE_PER_GPU=24

JOB_NAME=$MODEL

# NUM_NODES=2
# NUM_NODES=8
# NUM_NODES=32
# NUM_NODES=48
NUM_NODES=64
# NUM_NODES=128
# NUM_NODES=256

RESUME_FROM='' ## default
LOAD_FROM=''

###-----------------------------------------------------------------
CONFIG_FILE=configs/sapiens_mae/${DATASET}/${MODEL}.py

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

OUTPUT_DIR="Outputs/train/${DATASET}/${MODEL}/slurm"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"
CONDA_ENV='/uca/conda-envs/dgxenv-2023-09-25-7853/bin/activate'
TIME='7-00:00:00'
JOB_NAME="${JOB_NAME}"
WORLD_SIZE=$(($NUM_NODES * 8))

###-----------------------------------------------------------------
# Exporting variables
export MODEL NUM_NODES WORLD_SIZE TRAIN_BATCH_SIZE_PER_GPU CONFIG_FILE OPTIONS OUTPUT_DIR RSC_JOB_UUID
export TF_CPP_MIN_LOG_LEVEL=2

ROOT_DIR=$(cd ../../../.. && pwd)

## add this line if you wish to debug slurm
# #SBATCH --nodelist=avalearn1089,avalearn1090

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
# Print the script content in green
echo -e "\033[0;32m"
cat $SCRIPT
echo -e "\033[0m"

###-----------------------------------------------------------------
# Submit the job
sbatch $SCRIPT

###Optionally, remove the temporary script file
rm -f $SCRIPT
