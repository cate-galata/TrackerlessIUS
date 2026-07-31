#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p1          
#SBATCH --account=jkq@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --job-name=compute_metrics

module load miniforge/24.9.0
conda init
conda activate miccai24

cd $WORK/clean/TrackerlessIUS

# Compute metrics: protocol N1

CASE=Case011
python compute_metrics.py --case $CASE


CASE=Case025
python compute_metrics.py --case $CASE


CASE=Case045
python compute_metrics.py --case $CASE


CASE=Case052
python compute_metrics.py --case $CASE


CASE=Case056
python compute_metrics.py --case $CASE


CASE=Case070
python compute_metrics.py --case $CASE


CASE=Case085
python compute_metrics.py --case $CASE


CASE=Case103
python compute_metrics.py --case $CASE


CASE=Case112
python compute_metrics.py --case $CASE


CASE=Case114
python compute_metrics.py --case $CASE



# Compute metrics: protocol N2

CASE=Case027
python compute_metrics.py --case $CASE


CASE=Case045
python compute_metrics.py --case $CASE


CASE=Case074
python compute_metrics.py --case $CASE


CASE=Case085
python compute_metrics.py --case $CASE


CASE=Case099
python compute_metrics.py --case $CASE


CASE=Case103
python compute_metrics.py --case $CASE


CASE=Case112
python compute_metrics.py --case $CASE