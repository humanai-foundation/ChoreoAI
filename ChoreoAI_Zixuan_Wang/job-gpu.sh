#!/bin/bash
#SBATCH -A m4474
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

srun python3 train.py