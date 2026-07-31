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
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case025
FOLD=0
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case045
FOLD=1
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case052
FOLD=1
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case056
FOLD=1
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case070
FOLD=1
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case085
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case103
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case112
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case114
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE



METHOD=n2
K=1
SEED=2


CASE=Case027
FOLD=0
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case045
FOLD=1
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case074
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case085
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case099
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case103
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE


CASE=Case112
FOLD=2
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --output experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --case $CASE