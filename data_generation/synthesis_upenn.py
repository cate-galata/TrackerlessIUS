#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utilities.dataset import DatasetReMINDPred
from utilities.utils import (
    save,
    set_determinism)

from networks.mhvae import MHVAE2D

import time

def run_inference(paths_dict, saving_path, model, device, opt):

    # Define transforms for data normalization and augmentation

    subjects_dataset = DatasetReMINDPred(
        paths_unnorm=paths_dict, 
        normalization=True, 
        type_normalization=opt.type_normalization)
    dataloader = DataLoader(subjects_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    
    
    model.eval()  # Set model to evaluate mode

    torch.cuda.synchronize()
    start = time.perf_counter()

    # Iterate over data
    for batch in dataloader:
        
        imgs = dict()
        imgs_norm = dict()
        nonempty_list = []
        for mod in opt.modalities:
            if mod in batch.keys():
                # imgs[mod] = batch[mod].to(device).permute(0,4,1,2,3)
                imgs[mod] = batch[mod].to(device).permute(0,2,1,3,4)
                # print(f"DEBUG: Modality {mod} shape is {imgs[mod].shape}", flush=True)
                imgs[mod] = imgs[mod].reshape(-1, 1, *imgs[mod].shape[3:5])
                # print(f"DEBUG: Modality {mod} shape is {imgs[mod].shape}", flush=True)
                nonempty_list.append(mod)

        # print(f"Modalities: {nonempty_list}", flush=True)
        
        first_mod = nonempty_list[0]
        print('first_mod ', first_mod, flush=True)
        subset_mr = [k for k in nonempty_list if not 'us'==k]
        target_modality = model.modalities[0]
        print('target_mod ', target_modality, flush=True)
        with torch.no_grad():      
            affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
            name = batch[f"{first_mod}_name"][0].split('_')[0] + '_{}.nii.gz'
            
            temps = [0.3,0.5,0.7,1.]
            # temp = np.random.choice(temps)
            for temp in temps:
                # Save MR
                pred, _, _  = model({mod:imgs[mod].clone() for mod in subset_mr}, temp, return_feat=True, return_cat=True)
                pred = pred[:,0:1,...]
                save(pred, affine, os.path.join(saving_path, name.format(temp)))

    torch.cuda.synchronize()
    end = time.perf_counter() - start
    print(f'Time: {end:.3f}s')      
                        

def main():
    opt = parsing_data()

    set_determinism(seed=opt.seed)

    path_data = "/lustre/fsn1/projects/rech/jkq/ubt15jc/remind-100/upenn"

    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        
    # MODEL
    print("[INFO] Building hierarchical multi-modal model")
    model = MHVAE2D(
        modalities=opt.modalities,
        base_num_features=opt.base_features,
        num_pool=opt.pools,
        original_shape=opt.spatial_shape[:2],
        max_features=opt.max_features,
        with_residual=opt.no_res,
        with_se=opt.no_se,
        nb_finalblocks=opt.nb_finalblocks,
        nfeat_finalblock=opt.nfeat_finalblock,
        ).to(device)
    model.eval()

    print("[INFO] Reading data", flush=True)

    for folder in tqdm(os.listdir(path_data)):

        input_path = os.path.join(path_data, folder, 'data_mri')
        output_path = os.path.join(path_data, folder, 'data_us_old')

        if os.path.exists(output_path) and len(os.listdir(output_path)) >= 280:
            print(f"Case already processed, skipping.")
            continue
        else:
            os.makedirs(output_path, exist_ok=True)

        # PHASES
        nii_files = [k for k in os.listdir(input_path) if '.nii.gz' in k]

        from collections import defaultdict
        groups = defaultdict(dict)
        for fn in nii_files:
            grp = fn.split('_', 1)[0]                      # '00188-11-FLAIR-T1GD-T2'
            token = fn.rsplit('_', 1)[1].split('.')[0]     # 'T2' or 'seg'
            key = token.lower()
            if key == 'seg' :
                continue
            if key == 't1gd':
                key = 'cet1'
            path = os.path.join(input_path, fn) 
            groups[grp][key] = path
        # return list of dicts (sorted by group name)
        paths_dict = [groups[k] for k in sorted(groups)]

        # Training parameters
        fold = np.random.randint(0, 3)
        model_dir = os.path.join(opt.model_dir, f'mmhvae_f{fold}')
        save_path = os.path.join(model_dir, 'models', './CP_{}_{}.pth')
        assert os.path.exists(save_path.format('main',opt.epoch_inf)), f"Model weights not found: {save_path.format('main', opt.epoch_inf)}"

        model.load_state_dict(torch.load(save_path.format('main',opt.epoch_inf)))
        print(f"Loading model from {save_path.format('main',opt.epoch_inf)}", flush=True)

        run_inference(
            paths_dict, 
            output_path,
            model, 
            device, 
            opt)
        


def parsing_data():
    parser = argparse.ArgumentParser(
        description='Inference using MMHVAE')

    parser.add_argument('--model_dir',
                        type=str,
                        default='models/synthesis',
                        help='Save model directory')

    parser.add_argument('--input',
                        type=str,
                        help='Input folder')

    parser.add_argument('--output',
                        type=str,
                        help='Output folder')

    
    parser.add_argument('--type_normalization',
                        type=str,
                        default='min-max',
                        help='Type of normalization')
    
    parser.add_argument('--seed',
                    type=int,
                    default=3)
    
    parser.add_argument('--temp',
                    type=float,
                    default=0.7)

    parser.add_argument('--max_features',
                    type=int,
                    default=128,
                    help='Max Latent Dimensionality')

    parser.add_argument('--base_features',
                    type=int,
                    default=16,
                    help='Latent Dimensionality highest level (divided by 2)')
    
    parser.add_argument('--nb_finalblocks',
                    type=int,
                    default=6,
                    help='Number of blocks from z_1 to x_i')
    
    parser.add_argument('--nfeat_finalblock',
                    type=int,
                    default=8,
                    help='Number of channels in blocks from z_1 to x_i')

    parser.add_argument('--pools',
                    type=int,
                    default=6,
                    help='Number of latent representation below z_1')

    parser.add_argument('--epoch_inf',
                    type=int,
                    default=1000,
                    help='Epoch used for inference')

    parser.add_argument('--workers',
                    type=int,
                    default=10,
                    help='Number of workers')

    parser.add_argument('--spatial_shape',
                    type=int,
                    nargs="+",
                    default=(192,192))

    parser.add_argument('--modalities',
                    type=str,
                    nargs="+",
                    default=['us', 't2', 'cet1', 'flair'])

    parser.add_argument('--no_se',
                        action='store_false', 
                        help='Squeeze and Excitation disabled')

    parser.add_argument('--no_res', 
                        action='store_false', 
                        help='Residual connection disabled')
    
    parser.add_argument('--type_model', 
                        type=str,
                        default='mhvae',
                        )
    
    parser.add_argument('--save_images', 
                        action='store_true', 
                        help='Save images')
    
    parser.add_argument('--save_features', 
                        action='store_true', 
                        help='Save features')
    
    parser.add_argument('--case',
                        type=str,
                        default='Case003',
                        help='ID of the case')
    

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()