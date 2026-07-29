
# MICCAI 2024 Experiments Repository

This repository contains the code used to reproduce the experiments presented at **MICCAI 2024**.
It provides scripts to generate synthetic data, train patient-specific models, and perform inference.

---

## Workflow Overview

1. Generate synthetic **MR** sweeps
2. Generate synthetic **US** sweeps
3. Train a **patient-specific model**
4. Run **inference** on a single case

---

## Requirements

```bash

conda create -n miccai24 python=3.11
pip install -r requirements.txt
```

## 0. Set the experiment paramaters
Set the experiment parameters:

```bash
METHOD=remind
CASE=Case112
K=10
```

## 1. Generate Synthetic MR Sweeps

Generate (K = 10) synthetic MR sweeps:

```bash
python generate_data_new.py --K $K --annotator $METHOD --case $CASE
```

---

## 2. Generate Synthetic US Sweeps
Set the experiment parameters:

```bash
FOLD=2
```
Perform synthesis

```bash
python synthesis.py \
    --model_dir models/synthesis/mmhvae_f$FOLD \
    --input miccai2024_data/synthetic/$METHOD-$K/$CASE/data_mri/ \
    --output miccai2024_data/synthetic/$METHOD-$K/$CASE/data_us/ \
    --case $CASE
```

---

## 3. Train a Patient-Specific Model

Set the training parameters:

```bash
EPOCH=100
LR=0.01
```

Train the model:

```bash
python train.py \
    --model_dir ./models/ \
    --path_data ./miccai2024_data/synthetic/$METHOD-$K/$CASE/data_us/ \
    --path_labels ./miccai2024_data/synthetic/$METHOD-$K/$CASE/data_mri/ \
    --batch_size 1 \
    --learning_rate $LR \
    --epochs $EPOCH \
    --case $CASE \
    --comment $METHOD-$K-$EPOCH-$LR
```

Train the model with on the fly data generation:

```bash
python train_new.py \
    --model_dir ./models/ \
    --synthesizer_dir ./models/synthesis/mmhvae_f$FOLD \
    --path_data ../data/coregistered/mri-space \
    --path_strip ./miccai2024_data/skullstripping_hdbet/ \
    --batch_size 1 \
    --learning_rate $LR \
    --epochs $EPOCH \
    --case $CASE \
    --comment $METHOD-otf-$EPOCH-$LR
```

---

## 4. Perform Inference

Run inference on a single case using the trained model:

```bash
python inference.py \
    --model_dir ./models/ \
    --path_data_true ./miccai2024_data/test_set/${METHOD}/imgs/reslice${CASE}_crop.nii.gz \
    --batch_size 2 \
    --learning_rate $LR \
    --case $CASE \
    --spacial \
    --comment $METHOD-$K-$EPOCH-$LR 
```

---


