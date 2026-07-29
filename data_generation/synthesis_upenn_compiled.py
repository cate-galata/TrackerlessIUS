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

from networks.mmhvae import MHVAE2D, build_compiled_inference_fns


def run_inference(paths_dict, saving_path, model, encoders, decoders, device, opt):

    subjects_dataset = DatasetReMINDPred(
        paths_unnorm=paths_dict, 
        normalization=True, 
        type_normalization=opt.type_normalization)
    dataloader = DataLoader(subjects_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    model.eval()

    temps = [0.3, 0.5, 0.7, 1.]
    # temp is a tensor input (not a python float) so a single compiled
    # decoder graph serves every temperature value -- no recompilation
    # per temp.
    temp_tensors = [torch.tensor(t, device=device) for t in temps]

    for batch in dataloader:
        
        imgs = dict()
        nonempty_list = []
        for mod in opt.modalities:
            if mod in batch.keys():
                im = batch[mod].to(device).permute(0, 2, 1, 3, 4)
                im = im.reshape(-1, 1, *im.shape[3:5])
                imgs[mod] = im
                nonempty_list.append(mod)

        first_mod = nonempty_list[0]
        subset_mr = [k for k in nonempty_list if k != 'us']
        k = len(subset_mr)

        with torch.inference_mode():
            affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
            name = batch[f"{first_mod}_name"][0].split('_')[0] + '_{}.nii.gz'

            # Encode each present MR modality ONCE. This is temp-independent
            # (temp only affects the sampling step in decode), so unlike the
            # original script which reran the full forward pass (encoder
            # included) for every one of the 4 temperatures, the encoder now
            # only runs once per subject.
            x_list, skips_list = [], []
            for mod in subset_mr:
                x_mod = imgs[mod]
                torch._dynamo.mark_dynamic(x_mod, 0)  # N (slice count) varies per subject
                h, skips = encoders[mod](x_mod)
                x_list.append(h)
                skips_list.append(skips)

            mask = (imgs[subset_mr[0]] > -1).float()

            decode_fn = decoders[k]
            for temp, temp_t in zip(temps, temp_tensors):
                output_img = decode_fn(x_list, skips_list, temp_t, mask)
                pred = torch.cat([output_img[mod] for mod in model.modalities], 1)
                pred = pred[:, 0:1, ...]
                save(pred, affine, os.path.join(saving_path, name.format(temp)))
        
                        

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
    print("[INFO] Building hierarchical multi-modal model", flush=True)
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

    # Build + warm up the 6 compiled graphs ONCE, before the fold/subject
    # loop. This only depends on model structure (module shapes, which
    # modalities exist), not on parameter values -- so reloading weights
    # per-fold below via load_state_dict (which updates parameter data
    # in place) does not invalidate these compiled graphs or trigger a
    # recompile.
    print("[INFO] Compiling 3 per-modality encoders + 3 per-count decoders", flush=True)
    encoders, decoders = build_compiled_inference_fns(
        model,
        compile_mode="max-autotune-no-cudagraphs",  # switch to "default" if warmup time doesn't amortize for you
        device=device,
        spatial_shape=tuple(opt.spatial_shape[:2]),
    )

    print("[INFO] Reading data", flush=True)

    for folder in tqdm(os.listdir(path_data)):

        print(f"Processing {folder}...", flush=True)

        input_path = os.path.join(path_data, folder, 'data_mri')
        output_path = os.path.join(path_data, folder, 'data_us')

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
            encoders,
            decoders,
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