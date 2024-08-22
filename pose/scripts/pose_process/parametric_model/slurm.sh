#!/bin/bash

# NUM_NODES=64
NUM_NODES=14
# NUM_NODES=16

##-------------------------------------------------------------------------
RANGE_PER_NODE=$(( (END_INDEX - START_INDEX) / NUM_NODES ))

OUTPUT_DIR="/home/rawalk/Desktop/sapiens/pose/Outputs/pose_process/slurm"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"
CONDA_ENV='/uca/conda-envs/dgxenv-2024-07-04-07-29-05-x5807-centos9-py310-pt231'
TIME='7-00:00:00'
JOB_NAME="param_pose"
WORLD_SIZE=$(($NUM_NODES * 8))

export NUM_NODES WORLD_SIZE OUTPUT_DIR

ROOT_DIR=$(pwd)
mkdir -p "${OUTPUT_DIR}/slurm"

for ((i=0; i<NUM_NODES; i++))
do

    # Create a temporary script file
    SCRIPT=$(mktemp)

    echo $i
    FOLDER_NAME=$(printf "%03d" $i)  # Convert int i into zfill(3). eg: 000, 001, 002,... 010, 011, 012, 013.

#     # Write the SLURM script to the temporary file
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
echo "Processing folder: $FOLDER_NAME"
./slurm_helper.sh $FOLDER_NAME
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
