#!/bin/bash
#
#SBATCH --job-name=RNN1
#SBATCH --output=RNN1.out
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


python3 -u rnn.py --cuda --trainname train.txt --validname valid.txt --batchsize 20 --embedsize 200 --statesize 650
