#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH --account=jkq@a100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=synthesize_us
#SBATCH --qos=qos_gpu_a100-dev

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

METHOD=n1
K=1
SEED=2


CASE=Case011
FOLD=0
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case025
FOLD=0
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case045
FOLD=1
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case052
FOLD=1
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case056
FOLD=1
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case070
FOLD=1
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case085
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case103
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case112
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case114
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED



METHOD=n2
K=1
SEED=2


CASE=Case027
FOLD=0
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case045
FOLD=1
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case074
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case085
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case099
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case103
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED


CASE=Case112
FOLD=2
python synthesis.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED