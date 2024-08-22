#!/bin/bash

START_INDEX=0
# # END_INDEX=43604 ## true
# END_INDEX=43610 ## for safety

END_INDEX=800

# NUM_NODES=64
# NUM_NODES=8
NUM_NODES=16

##-------------------------------------------------------------------------
RANGE_PER_NODE=$(( (END_INDEX - START_INDEX) / NUM_NODES ))

OUTPUT_DIR="/home/rawalk/Desktop/sapiens/pose/Outputs/pose_process/slurm"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"
CONDA_ENV='/uca/conda-envs/dgxenv-2023-09-25-7853/bin/activate'
TIME='7-00:00:00'
JOB_NAME="cca_pose"
WORLD_SIZE=$(($NUM_NODES * 8))

export NUM_NODES WORLD_SIZE OUTPUT_DIR

ROOT_DIR=$(pwd)
mkdir -p "${OUTPUT_DIR}/slurm"

for ((i=0; i<NUM_NODES; i++))
do
    START_SAMPLE_ID=$(( START_INDEX + i * RANGE_PER_NODE ))
    END_SAMPLE_ID=$(( START_SAMPLE_ID + RANGE_PER_NODE ))

    if [[ $i -eq $((NUM_NODES - 1)) ]]; then
        END_SAMPLE_ID=$END_INDEX
    fi

    # Create a temporary script file
    SCRIPT=$(mktemp)

    # Write the SLURM script to the temporary file
    cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=learn
#SBATCH --time=$TIME
#SBATCH --job-name=${JOB_NAME}_${i}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --output=${OUTPUT_DIR}/slurm/%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm/%j.err
cd $ROOT_DIR
source ${CONDA_ENV}
echo "Processing range: $START_SAMPLE_ID to $END_SAMPLE_ID"
./slurm_helper.sh $START_SAMPLE_ID $END_SAMPLE_ID
EOL

    # Display the script in green
    echo -e "\033[0;32m"
    cat $SCRIPT
    echo -e "\033[0m"

    # # Submit the job
    sbatch $SCRIPT

    # Optionally, remove the temporary script file
    rm -f $SCRIPT
done

echo "All jobs submitted."
