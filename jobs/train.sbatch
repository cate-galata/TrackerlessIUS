#!/bin/bash
#SBATCH --time 06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH --account=jkq@a100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=train
#SBATCH --qos=qos_gpu_a100-t3

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

CASE=Case011
METHOD=n1
K=10
EPOCH=100
LR=0.01
SEED=2

python train.py \
    --model_dir ./models/ \
    --path_data ./experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_us/ \
    --path_labels ./experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_mri/ \
    --batch_size 1 \
    --learning_rate $LR \
    --epochs $EPOCH \
    --case $CASE \
    --comment $METHOD-$K-$EPOCH-$LR-$SEED