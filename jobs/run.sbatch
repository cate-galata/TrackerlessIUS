#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C h100
#SBATCH --account=jkq@h100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=run
#SBATCH --qos=qos_gpu_h100-dev

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

METHOD=n1
K=10
EPOCH=100
LR=0.01
CASE=Case103
FOLD=2
SEED=2

python generate_and_synthesize.py --K $K --annotator $METHOD --case $CASE --fold $FOLD --seed $SEED

# python train.py \
#     --model_dir ./models/ \
#     --path_data ./experiments/synthetic/$METHOD-$K-$SEED/$CASE/synthetic_data_reslice/ \
#     --path_labels ./experiments/synthetic/$METHOD-$K-$SEED/$CASE/data_reslice/ \
#     --batch_size 1 \
#     --learning_rate $LR \
#     --epochs $EPOCH \
#     --case $CASE \
#     --seed $SEED \
#     --comment $METHOD-$K-$EPOCH-$LR-$SEED

# python inference.py \
#     --model_dir ./models/ \
#     --path_data_true ./experiments/test_set/${METHOD}/imgs/reslice${CASE}_crop.nii.gz \
#     --batch_size 1 \
#     --learning_rate $LR \
#     --case $CASE \
#     --seed $SEED \
#     --comment $METHOD-$K-$EPOCH-$LR-$SEED