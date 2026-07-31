#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH --account=jkq@a100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=generate_sweep
#SBATCH --qos=qos_gpu_a100-dev

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

METHOD=n1
K=1
SEED=2

CASE=Case011
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case025
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case045
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case052
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case056
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case070
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case085
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case103
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case112
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case114
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


METHOD=n2
K=1
SEED=2


CASE=Case027
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case045
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case074
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case085
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case099
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case103
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED


CASE=Case112
python generate_data.py --K $K --annotator $METHOD --case $CASE --seed $SEED
python generate_data_new.py --K $K --annotator $METHOD --case $CASE --seed $SEED
