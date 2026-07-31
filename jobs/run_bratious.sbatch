#!/bin/bash
#SBATCH --time 00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C a100
#SBATCH --account=jkq@a100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=run_bratious
#SBATCH --qos=qos_gpu_a100-dev

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

METHOD=n1
EPOCH=100
LR=0.01

python train_bratious.py \
    --model_dir ./models/ \
    --path_data ./data/bratious/images/ \
    --path_labels ./data/bratious/labels/ \
    --batch_size 4 \
    --learning_rate $LR \
    --epochs $EPOCH \
    --comment $METHOD-$EPOCH-$LR

CASE=Case011

python inference_bratious.py \
    --model_dir ./models/ \
    --path_data_true ./experiments/test_set/${METHOD}/imgs/reslice${CASE}_crop.nii.gz \
    --batch_size 1 \
    --learning_rate $LR \
    --case $CASE \
    --comment $METHOD-$EPOCH-$LR

CASE=Case025

python inference_bratious.py \
    --model_dir ./models/ \
    --path_data_true ./experiments/test_set/${METHOD}/imgs/reslice${CASE}_crop.nii.gz \
    --batch_size 1 \
    --learning_rate $LR \
    --case $CASE \
    --comment $METHOD-$EPOCH-$LR