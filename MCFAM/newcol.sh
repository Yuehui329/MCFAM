#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
module load anaconda/anaconda3-2022.10
module load cuda/11.7.0
source activate DL
#python drugbank/main.py --cfg conf.pyDTI
python main.py