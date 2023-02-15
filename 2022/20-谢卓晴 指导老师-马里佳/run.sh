#!/bin/bash

#SBATCH --job-name=SENTIMENT_PERIN
#SBATCH --account=nn9851k
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus=1

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-pytorch/1.6.0-gomkl-2019b-cuda-10.1.243-Python-3.7.4
module load nlpl-transformers/4.5.1-gomkl-2019b-Python-3.7.4
module load nlpl-scipy-ecosystem/2021.01-gomkl-2019b-Python-3.7.4
module load sentencepiece/0.1.94-gomkl-2019b-Python-3.7.4
module load nlpl-nltk/3.5-gomkl-2019b-Python-3.7.4

TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline python3 train.py --log_wandb --config "$1"
