#!/bin/bash
#
#SBATCH --job-name=fasttextModels
#SBATCH --output=fasttextModels.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module purge
module load python3/intel
module load cuda
module load cudnn
module load pytorch


python3 -u fastTextModels.py
