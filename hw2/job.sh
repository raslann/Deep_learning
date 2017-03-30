#!/bin/bash
#
#SBATCH --job-name=rnn
#SBATCH --output=rnn.out
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


python3 -u rnn.py --trainname ptb/train.txt --validname ptb/valid.txt --cuda
