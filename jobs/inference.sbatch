#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH --account=jkq@a100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=inference
#SBATCH --qos=qos_gpu_a100-dev

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

python inference.py \
    --model_dir ./models/ \
    --path_data_true ./experiments/test_set/${METHOD}/imgs/reslice${CASE}_crop.nii.gz \
    --batch_size 2 \
    --learning_rate $LR \
    --case $CASE \
    --comment $METHOD-$K-$EPOCH-$LR-$SEED

