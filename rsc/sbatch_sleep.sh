#!/bin/sh
#SBATCH --partition=learn
#SBATCH --time=7-00:00:00
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --error=/home/%u/logs/job.%J.err
#SBATCH --output=/home/%u/logs/job.%J.out
#SBATCH --qos=gen_ca  # Use the QOS

srun sleep 7d
