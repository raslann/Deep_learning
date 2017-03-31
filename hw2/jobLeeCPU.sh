#!/bin/bash
#
#SBATCH --job-name=charcpu
#SBATCH --output=charcpu.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00

module purge
module load python3/intel
module load pytorch


python3 -u CharacterAwareModel.py --trainname ptb/train.txt --validname ptb/valid.txt
