#!/bin/bash
#SBATCH --time 02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C h100
#SBATCH --account=jkq@h100
#SBATCH --hint=nomultithread 
#SBATCH --job-name=run_otf
#SBATCH --qos=qos_gpu_h100-dev

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

CASE=Case103
METHOD=n1
EPOCH=100
LR=0.01
FOLD=2
SEED=2
DROP=cran

python train_otf.py \
    --model_dir ./models/ \
    --synthesizer_dir ./models/synthesis/mmhvae_f$FOLD \
    --path_data ./data/coregistered/mri-space \
    --path_strip ./experiments/skullstripping_hdbet/ \
    --batch_size 1 \
    --learning_rate $LR \
    --epochs $EPOCH \
    --case $CASE \
    --output_val ./experiments/synthetic/$METHOD-validation_otf-$SEED \
    --comment $METHOD-otf-$EPOCH-$LR-$SEED \
    --seed $SEED \
    --drop_modalities $DROP

python inference.py \
    --model_dir ./models/ \
    --path_data_true ./experiments/test_set/${METHOD}/imgs/reslice${CASE}_crop.nii.gz \
    --batch_size 1 \
    --learning_rate $LR \
    --case $CASE \
    --comment $METHOD-otf-$EPOCH-$LR-$SEED \
    --seed $SEED