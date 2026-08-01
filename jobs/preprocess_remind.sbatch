#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p1          
#SBATCH --account=jkq@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --job-name=preprocess_data

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

python create_dataset_mri-space.py 
python create_dataset_us-space.py 