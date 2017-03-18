#!/bin/bash
#
#SBATCH --job-name=ladder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module purge
. ~/scripts/modules-mxnet.sh
python -u rnn.py --cuda --trainname train.txt --validname valid.txt --batchsize 20 --embedsize 200 --statesize 650
