import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import glob
import nibabel as nib
import torch
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from itertools import chain, combinations
from utilities.generatesweep import *
from utilities.generation import *
from utilities.utils import (
    save,
    set_determinism)
from networks.mhvae import MHVAE2D
from tqdm import tqdm

LOWER = 0.
UPPER = 99.95

def generate_us_sweep(total_subsets_mr, ultrasounds, filtered_arr, weights, TUMOR_POINTS_3D, m_2, surface_tree, mod_vols, device, output_path, radius_mm, Y_grid=None, X_grid=None, mode='training', case=None, video_clip=False):
    modalities_set = random.choice(total_subsets_mr)
    us = next(ultrasounds)

    chosen_idx = np.random.choice(filtered_arr.shape[0], p=weights)
    P_0 = filtered_arr[chosen_idx]

    # Load US + probe mask + keypoints
    kp_img = nib.load(f'../data/precomputed_us_masks/{us}/{us}-keypoints.nii.gz')
    keypoints = kp_img.get_fdata()
    frames_w_surface = np.argwhere(keypoints==4)[:, 2]

    fov_img = nib.load(f'../data/precomputed_us_masks/{us}/{us}-fov_mask.nii.gz')
    fov_mask = fov_img.get_fdata()
    fov_mask = torch.from_numpy(fov_mask).to(torch.float32).to(device)

    z_mid = np.random.choice(frames_w_surface)

    fov_slice = fov_mask[...,z_mid]
    points_mask = keypoints[...,z_mid]
    ys, xs = np.where(points_mask > 0)
    labels = points_mask[ys, xs]

    idx = labels == 4
    P_us = [xs[idx][0], ys[idx][0]]
    P_us = torch.from_numpy(np.array(P_us)).to(torch.float32).to(device)

    # --------------------
    # 5) Determine how to scan
    # --------------------
    # Define d_vec_0 pointing from P_0 to m_2
    d_vec_0 = m_2 - P_0
    d_vec_0 = d_vec_0 / np.linalg.norm(d_vec_0)

    # Select the sweep direction
    n = d_vec_0 
    min_idx = np.argmin(np.abs(n))
    v = np.zeros(3)
    v[min_idx] = 1
    E1 = np.cross(n, v)
    E1 = E1 / np.linalg.norm(E1) 
    E2 = np.cross(n, E1) 

    TUMOR_POINTS_2D = []

    for T in TUMOR_POINTS_3D:
        T_minus_P0 = T - P_0
        x_prime = np.dot(T_minus_P0, E1)
        y_prime = np.dot(T_minus_P0, E2)
        TUMOR_POINTS_2D.append([x_prime, y_prime])
        
    TUMOR_POINTS_2D = np.array(TUMOR_POINTS_2D)

    pca = PCA(n_components=2)
    pca.fit(TUMOR_POINTS_2D) 

    variance = pca.explained_variance_
    components = pca.components_

    PC1_3D_on_plane = (components[0, 0] * E1) + (components[0, 1] * E2)
    PC1_3D_on_plane /= np.linalg.norm(PC1_3D_on_plane)

    PC2_3D_on_plane = (components[1, 0] * E1) + (components[1, 1] * E2)
    PC2_3D_on_plane /= np.linalg.norm(PC2_3D_on_plane)

    x = np.random.normal(0, np.sqrt(variance[0]))
    y = np.random.normal(0, np.sqrt(variance[1]))

    sweep_dir = (x * PC1_3D_on_plane) + (y * PC2_3D_on_plane)
    sweep_dir /= np.linalg.norm(sweep_dir)

    # Define i_vec_0 orthogonal to d_vec_0 and sweep_dir
    i_vec_0 = np.cross(d_vec_0, sweep_dir)
    i_vec_0 = i_vec_0 / np.linalg.norm(i_vec_0)

    # Find min and max coordinates of tumor along sweep_dir
    T_prime = TUMOR_POINTS_3D - P_0

    TUMOR_POINTS_1D = np.dot(T_prime, sweep_dir)

    tumor_min = np.min(TUMOR_POINTS_1D)
    tumor_max = np.max(TUMOR_POINTS_1D)

    tumor_min_3D = P_0 + (tumor_min * sweep_dir) 
    tumor_max_3D = P_0 + (tumor_max * sweep_dir)

    # add probe angle around P_0
    Ps = [P_0]
    d_vecs = []
    i_vecs = []
    num_steps = 100

    for i in range(num_steps):
        t = i / (num_steps - 1)   
        if t > 0.25:
            break

        d_vec = slerp(d_vec_0, sweep_dir, t)
        d_vec /= np.linalg.norm(d_vec)

        d_vecs.append(d_vec)
        i_vecs.append(i_vec_0)

    Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i-1, axis=0)])

    d_vecs += d_vecs[::-1]
    i_vecs += i_vecs[::-1]

    Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

    d_vecs_init = []
    i_vecs_init = []

    for i in range(num_steps):
        t = i / (num_steps - 1)   
        if t > 0.25:
            break

        d_vec = slerp(d_vec_0, -sweep_dir, t)
        d_vec /= np.linalg.norm(d_vec)

        d_vecs_init.append(d_vec)
        i_vecs_init.append(i_vec_0)

    Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

    d_vecs_init += d_vecs_init[::-1]
    i_vecs_init += i_vecs_init[::-1]

    Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

    d_vecs = d_vecs_init + d_vecs
    i_vecs = i_vecs_init + i_vecs

    # Sweep along sweep_dir updating Ps and keeping the image plane constant 
    sample_Ps, Ps_before, Ps_after = sample_points_along_straight_line(P_0, sweep_dir, tumor_min, tumor_max)

    Ps_after = walk_on_surface(
        filtered_arr,
        x0_world=P_0,
        n_world=sweep_dir,
        dx_mm=0.5,
        n_steps=len(Ps_after),
        len_max_mm=np.abs(tumor_max),
        radius_mm=radius_mm,
    )

    Ps_before = walk_on_surface(
        filtered_arr,
        x0_world=P_0,
        n_world=-sweep_dir,
        dx_mm=0.5,
        n_steps=len(Ps_before),
        len_max_mm=np.abs(tumor_min),
        radius_mm=radius_mm,
    )
    Ps_before = Ps_before[::-1][1:]

    d_vecs_before = [d_vec_0 for _ in range(Ps_before.shape[0])]
    i_vecs_before = [i_vec_0 for _ in range(Ps_before.shape[0])]
    d_vecs_after = [d_vec_0 for _ in range(Ps_after.shape[0])]
    i_vecs_after = [i_vec_0 for _ in range(Ps_after.shape[0])]

    Ps = np.vstack([Ps_before, Ps, Ps_after])
    d_vecs = d_vecs_before + d_vecs + d_vecs_after
    i_vecs = i_vecs_before + i_vecs + i_vecs_after

    # Add angles to image plane at the start and end of the sweep
    num_steps = 100

    for i in range(num_steps):
        t = i / (num_steps - 1)   
        if t > 0.25:
            break

        d_vec = slerp(d_vec_0, sweep_dir, t)
        d_vec /= np.linalg.norm(d_vec)

        d_vecs.append(d_vec)
        i_vecs.append(i_vec_0)

    Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

    d_vecs_init = []
    i_vecs_init = []

    for i in range(num_steps):
        t = i / (num_steps - 1)   
        if t > 0.25:
            break

        d_vec = slerp(d_vec_0, -sweep_dir, t)
        d_vec /= np.linalg.norm(d_vec)

        d_vecs_init.append(d_vec)
        i_vecs_init.append(i_vec_0)

    d_vecs_init.reverse()
    i_vecs_init.reverse()

    d_vecs = d_vecs_init + d_vecs
    i_vecs = i_vecs_init + i_vecs

    Ps = np.vstack([np.repeat(Ps[0][None, :], i, axis=0), Ps])

    arr_Ps = np.array(Ps)
    arr_d_vecs = np.array(d_vecs)
    arr_i_vecs = np.array(i_vecs)

    arr_Ps_torch = torch.as_tensor(arr_Ps).to(torch.float32)
    arr_d_vecs_torch = torch.as_tensor(arr_d_vecs).to(torch.float32)
    arr_i_vecs_torch = torch.as_tensor(arr_i_vecs).to(torch.float32)

    if video_clip == True:
        N = arr_Ps_torch.shape[0]
        num_points = 5
        step = 5

        # Calculate the maximum possible starting index
        max_start = N - ((num_points - 1) * step)

        if max_start <= 0:
            raise ValueError(f"Tensor is too short (N={N}) to sample {num_points} points with step {step}")

        start_idx = torch.randint(0, max_start, (1,)).item()
        indices = torch.arange(start_idx, start_idx + (num_points * step), step)

        # Resulting shape: [num_points, H, W]
        arr_Ps_torch = arr_Ps_torch[indices]          
        arr_d_vecs_torch = arr_d_vecs_torch[indices]
        arr_i_vecs_torch = arr_i_vecs_torch[indices]


    volume_3d_np = dict()
    volume_3d = dict()

    for mod in modalities_set+('target',):
        vol_t = mod_vols[mod]['volume'].to(device)
        inv_affine = mod_vols[mod]['inv_affine'].to(device)

        interp_mode = "nearest" if mod == "target" else "bilinear" 

        volume_slices_np = []
        volume_slices = []

        Ps = arr_Ps_torch.to(device)
        d_vecs = arr_d_vecs_torch.to(device)
        i_vecs = arr_i_vecs_torch.to(device) 

        slices = slice_pose_torch_batched(
            vol_t,
            inv_affine,
            Ps,
            d_vecs,
            i_vecs,
            dx_mm=0.5,
            dy_mm=0.5,
            H_out=192,
            W_out=192,
            Y=Y_grid, 
            X=X_grid, 
            P_origin=P_us,
            fov_mask=fov_slice,
            mode=interp_mode
        )

        volume_3d[mod] = slices.permute(1,2,0).unsqueeze(0)

        if mode != 'training':
            volume_3d_np[mod] = slices.detach().cpu().numpy().transpose(1, 2, 0)
            listToStr = "-".join([str(elem) for elem in modalities_set])
            flnm = f"{case}-{us}-{listToStr}"
            
            nib.save(
                nib.Nifti1Image(volume_3d_np[mod], mod_vols[mod]['affine']),
                os.path.join(output_path, flnm + f"_{mod}.nii.gz")
            )

    return volume_3d, modalities_set, us


def preprocess_mr_torch(x, k=[0,0,0], norm=True, type_normalization='standardization', mode='training'):
    if mode == 'training':
        if k[0] == 1:
            x = torch.flip(x, [0])
        if k[1] == 1:
            x = torch.flip(x, [1])
        if k[2] == 1:
            x = torch.flip(x, [2])
        x = x.clone()

    mask = x > 0
    x = x.float()

    if norm:
        if torch.any(mask):
            if type_normalization == 'standardization':
                max_data = torch.quantile(x, UPPER/100)
                x[x > max_data] = max_data
                sub = torch.mean(x)
                div = 3 * torch.std(x)
                x = (x - sub) / (div + 1e-8)
                x[x > 1] = 1
                x[~mask] = -1
            elif type_normalization == 'min-max':
                min_data = torch.quantile(x[mask], LOWER)
                max_data = torch.quantile(x[mask], UPPER/100)

                x[x>max_data] = max_data
                x = (x-min_data) / (max_data-min_data)
                x = x* (1 + 255/256) - 255 / 256
                
                x[~mask] = -1.
                div = (max_data - min_data) / 2
                sub = min_data
            else:
                raise NotImplementedError(f"Normalization {type_normalization} should be in : min-max, standardization")
        else:
            x -= 1
            div = 1
            sub = 1
    else:
        if torch.any(mask):
            x = 2*x - 1
            div = 1/2
            sub = 1/2          
        else:
            x -= 1
            div = 1
            sub = 1
    return x, sub, div


def synthesize_us_sweep(synthesizer, modalities, output, device, type_normalization, saving_path='', mode='training'):
  
    synthesizer.eval()  # Set synthesizer to evaluation mode

    imgs = dict()
    nonempty_list = []
    for mod in modalities:
        img, sub, div = preprocess_mr_torch(output[mod].squeeze(), norm=True, type_normalization=type_normalization, mode=mode)
        output[mod] = img.unsqueeze(0)

        output[mod] = output[mod].unsqueeze(0)
        imgs[mod] = output[mod].permute(0,4,1,2,3)
        imgs[mod] = imgs[mod].reshape(-1, 1, *output[mod].shape[2:4])
        nonempty_list.append(mod)
    
    first_mod = nonempty_list[0]
    subset_mr = [k for k in nonempty_list if not 'us'==k]
    with torch.no_grad():   
        temps = [0.3,0.5,0.7,1.]
        temp = np.random.choice(temps)

        pred, _, _  = synthesizer({mod:imgs[mod].clone() for mod in subset_mr}, temp, return_feat=True, return_cat=True)

        pred = pred[:,0:1,...]
        if mode != 'training':
            affine = np.eye(4)
            save(pred, affine, saving_path.format(temp))

    return pred