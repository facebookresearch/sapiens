#!/bin/sh
#SBATCH --partition=learn
#SBATCH --time=7-00:00:00
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --error=/home/rawalk/logs/job.%J.err
#SBATCH --output=/home/rawalk/logs/job.%J.out
#SBATCH --nodelist=avalearn1762  # Request specific node

srun sleep 7d
