import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import glob
import random
import math
import time
import ants
from scipy import ndimage
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation
import argparse
from itertools import chain, combinations
from utilities.generatesweep import *
from utilities.generation import *
import itertools

parser = argparse.ArgumentParser(
                    prog='Generatedata',
                    description='This script create sweeps of MR')

parser.add_argument('--K', type=int, default=1, help='Number of K sweeps')
parser.add_argument('--annotator', type=str, default="remind", help='Annotator')
parser.add_argument('--case', type=str, default="Case027", help='Annotator')


args = parser.parse_args()

# --------------------
# Hyperparameters
# --------------------
NB_cases = args.K
annotator = args.annotator
case = args.case

path_data = f'/lustre/fswork/projects/rech/jkq/ubt15jc/ds/upenn-gbm/UPENN-GBM/NIfTI-files/images_structural'
path_automated_segm = '/lustre/fswork/projects/rech/jkq/ubt15jc/ds/upenn-gbm/UPENN-GBM/NIfTI-files/automated_segm'
path_segm = '/lustre/fswork/projects/rech/jkq/ubt15jc/ds/upenn-gbm/UPENN-GBM/NIfTI-files/images_segm'
# path_output = "/lustre/fsn1/projects/rech/jkq/ubt15jc/upenn/{}/{}/{}"
path_output = '/lustre/fsn1/projects/rech/jkq/ubt15jc/synthetic_mri-us/{}/{}/{}'
us_path_folder = f'../data/coregistered/us-space'
reg_folder = '/lustre/fswork/projects/rech/jkq/ubt15jc/ds/upenn-gbm/UPENN-GBM/NIfTI-files/registration/mni'

mni_surf_path = '../data/mni/mni_brain_surface.nii.gz.seg.nrrd'
mni_non_viable_path = '../data/mni/refined_non_viable_surface_mask2.nii.gz'

np.random.seed(0)

H_out=192
W_out=192
device = torch.device("cuda:0")

ys = torch.arange(H_out, device=device, dtype=torch.float32)
xs = torch.arange(W_out, device=device, dtype=torch.float32)

Y_grid, X_grid = torch.meshgrid(ys, xs, indexing="ij")

# --------------------
# 1) Reference MR and segmentation mask
# --------------------
# MR modalities + all non-empty subsets
for folder in tqdm(os.listdir(path_data)):

    if os.path.exists(os.path.join("/lustre/fsn1/projects/rech/jkq/ubt15jc/synthetic_mri-us", folder)) and len(os.listdir(os.path.join("/lustre/fsn1/projects/rech/jkq/ubt15jc/synthetic_mri-us", folder, 'data_mri'))) >= 24:
        print(f"Case already processed, skipping.")
        continue

    id = folder.split('-')[-1]
    case = id.split('_')[0]
    scan = id.split('_')[1]

    
    images_modalities = [f.split("_")[-1].split(".")[0] for f in os.listdir(os.path.join(path_data, folder)) if f.endswith(".nii.gz")]
    modalities = [m for m in images_modalities if m != 'T1']
    # modalities = [k[0] for k in images_modalities]
    total_subsets_mr = list(
        chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1))
    )

    seg_candidates = glob.glob(os.path.join(path_segm, f'{folder}**.nii.gz'))
    if not seg_candidates:
        seg_candidates = glob.glob(os.path.join(path_automated_segm, f'{folder}**.nii.gz'))
        if not seg_candidates:
            print(f"[INFO] segmentation not found for {folder}, skipping.")
            continue
    path_target = seg_candidates[0]

    mod_vols = {}
    for mod in modalities + ['target']:
        if mod == 'target':
            mod = 'seg'
            path = path_target
        else:
            path = glob.glob(os.path.join(path_data, folder, f'{folder}_{mod}**.nii.gz'))[0]
        img = nib.load(path)
        data = img.get_fdata() 
        inv_affine = torch.linalg.inv(torch.from_numpy(img.affine))
        vol_np = np.transpose(data, (2, 1, 0))
        vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)
        mod_vols[mod] = {
            'data': data,
            'volume': vol_t,
            'affine': img.affine,
            'inv_affine': inv_affine
        }

    data_target = mod_vols['seg']['data']

    # --------------------
    # 2) Prepare output folders + candidate US list
    # --------------------
    for dir in ["data_mri"]:
        os.makedirs(path_output.format(folder, dir, ""), exist_ok=True)

    ultrasounds = [k for k in os.listdir(us_path_folder) if "Case" in k]

    # --------------------
    # 3) Determine where to scan
    # --------------------

    data_noncerebrum = (mod_vols["T1GD"]['data'] == 0)
    border = find_boundaries(data_noncerebrum, mode="inner")

    border_pixels = np.stack(np.where(border),-1)
    new_arr = vox_to_world_many(mod_vols['seg']['affine'], border_pixels)

    affine_path = os.path.join(reg_folder, folder, f"mri-to-mni-Syn0GenericAffine.mat")
    sitk_affine = sitk.ReadTransform(affine_path)

    mni_seg_img = sitk.ReadImage(mni_surf_path)
    mni_surface_mask = sitk.GetArrayFromImage(mni_seg_img)[..., 0] > 0 # (z,y,x)

    non_viable_surface_mask = sitk.GetArrayFromImage(sitk.ReadImage(mni_non_viable_path)) > 0 # (z,y,x)
    viable_surface_mask = mni_surface_mask & (~non_viable_surface_mask)
    non_viable_surface_mask_dilated = binary_dilation(non_viable_surface_mask, iterations=5)
    non_viable_surface_voxels = np.argwhere(non_viable_surface_mask_dilated) # MNI voxel space

    non_viable_surface_mri_world = []
    for idx in non_viable_surface_voxels:
        z, y, x = idx
        physical = mni_seg_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        surface_world = sitk_affine.TransformPoint(physical)
        non_viable_surface_mri_world.append([-1, -1, 1] * np.array(surface_world))

    non_viable_surface_mri_world = np.array(non_viable_surface_mri_world)

    tree = cKDTree(non_viable_surface_mri_world)

    img_mri = sitk.ReadImage(glob.glob(os.path.join(path_data, folder, f'{folder}_T2**.nii.gz'))[0])
    radius_mm = np.sqrt(img_mri.GetSpacing()[0]**2 + img_mri.GetSpacing()[1]**2 + img_mri.GetSpacing()[2]**2)

    idx = tree.query_ball_point(new_arr, r=radius_mm) # for every point in new_arr, idx returns the points in the tree which are within distance r
    mask = np.array([len(n) == 0 for n in idx])
    filtered_arr = new_arr[mask]

    surface_tree = cKDTree(filtered_arr) if len(filtered_arr) > 0 else None

    # Compute tumor centroid and its distance from the brain surface
    target_mask = np.argwhere(data_target > 0)
    target_world = vox_to_world_many(mod_vols['seg']['affine'], target_mask)

    TUMOR_POINTS_3D = np.array(target_world)

    m_2 = [k.mean() for k in np.where(data_target > 0)]
    m_2 += np.random.normal(loc=0.0, scale=3.0, size=(3))
    m_2 = vox_to_world(mod_vols['seg']['affine'], m_2)

    d_2 = np.linalg.norm(filtered_arr - m_2, axis=1)
    weights = np.exp(-d_2)
    weights[weights<0] = 0
    weights /= weights.sum()

    path_output_data = path_output.format(folder, "data_mri", "")

    # --------------------
    # 4) For each modality subset: generate NB_cases synthetic pairs (MR->US)
    # --------------------
    for modalities_set in total_subsets_mr:

        ultrasounds_subset = iter(np.random.choice(ultrasounds, NB_cases + 10, replace=False).tolist())
        print(modalities_set, flush=True)

        count = 0
        while count < NB_cases:
            us = next(ultrasounds_subset)
            # Sample initial probe position
            chosen_idx = np.random.choice(filtered_arr.shape[0], p=weights)
            P_0 = filtered_arr[chosen_idx]

            # Load US + probe mask + keypoints
            kp_img = nib.load(f'../data/precomputed_us_masks/{us}/{us}-keypoints.nii.gz')
            keypoints = kp_img.get_fdata()
            frames_w_surface = np.argwhere(keypoints==4)[:, 2]

            fov_img = nib.load(f'../data/precomputed_us_masks/{us}/{us}-fov_mask.nii.gz')
            fov_mask = fov_img.get_fdata()
            fov_mask = torch.from_numpy(fov_mask).to(torch.float32).to(device)

            # z_mid = fov_mask.shape[-1] // 2
            z_mid = np.random.choice(frames_w_surface)

            points_mask = keypoints[..., z_mid]
            fov_slice = fov_mask[..., z_mid]
            ys, xs = np.where(points_mask > 0)
            labels = points_mask[ys, xs]

            idx = labels == 4
            P_us = np.array([xs[idx][0], ys[idx][0]])  
            P_us_torch = torch.from_numpy(P_us).to(torch.float32).to(device)

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

            # Ps = [P_0]
            # d_vecs = []
            # i_vecs = []
            # num_steps = 100

            # for i in range(num_steps):
            #     t = i / (num_steps - 1)   
            #     if t > 0.25:
            #         break

            #     d_vec = slerp(d_vec_0, sweep_dir, t)
            #     d_vec /= np.linalg.norm(d_vec)

            #     d_vecs.append(d_vec)
            #     i_vecs.append(i_vec_0)

            # Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i-1, axis=0)])

            # d_vecs += d_vecs[::-1]
            # i_vecs += i_vecs[::-1]

            # Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

            # d_vecs_init = []
            # i_vecs_init = []

            # for i in range(num_steps):
            #     t = i / (num_steps - 1)   
            #     if t > 0.25:
            #         break

            #     d_vec = slerp(d_vec_0, -sweep_dir, t)
            #     d_vec /= np.linalg.norm(d_vec)

            #     d_vecs_init.append(d_vec)
            #     i_vecs_init.append(i_vec_0)

            # Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

            # d_vecs_init += d_vecs_init[::-1]
            # i_vecs_init += i_vecs_init[::-1]

            # Ps = np.vstack([Ps, np.repeat(Ps[-1][None, :], i, axis=0)])

            # d_vecs = d_vecs_init + d_vecs
            # i_vecs = i_vecs_init + i_vecs

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

            d_vecs_before = [d_vec_0] * Ps_before.shape[0]
            i_vecs_before = [i_vec_0] * Ps_before.shape[0]
            d_vecs_after = [d_vec_0] * Ps_after.shape[0]
            i_vecs_after = [i_vec_0] * Ps_after.shape[0]

            # Ps = np.vstack([Ps_before, Ps, Ps_after])
            # d_vecs = d_vecs_before + d_vecs + d_vecs_after
            # i_vecs = i_vecs_before + i_vecs + i_vecs_after

            Ps = np.vstack([Ps_before, Ps_after])
            d_vecs = d_vecs_before + d_vecs_after
            i_vecs = i_vecs_before + i_vecs_after

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

            arr_Ps_torch = torch.as_tensor(arr_Ps).to(torch.float32).to(device)
            arr_d_vecs_torch = torch.as_tensor(arr_d_vecs).to(torch.float32).to(device)
            arr_i_vecs_torch = torch.as_tensor(arr_i_vecs).to(torch.float32).to(device)

            listToStr = "-".join([str(elem) for elem in modalities_set])
            flnm = f"{case}-{scan}-{us}-{listToStr}"

            # Resample MR modalities into US space
            for mod in modalities_set: #+ ('seg',):
                vol_t = mod_vols[mod]['volume'].to(torch.float32).to(device)
                inv_affine = mod_vols[mod]['inv_affine'].to(torch.float32).to(device)

                interp_mode = "nearest" if mod == "seg" else "bilinear" 
                
                slices = slice_pose_torch_batched(
                        vol_t, inv_affine,
                        arr_Ps_torch, arr_d_vecs_torch, arr_i_vecs_torch,
                        dx_mm=0.5, dy_mm=0.5,
                        H_out=192, W_out=192,
                        Y=Y_grid, X=X_grid,
                        P_origin=P_us_torch,
                        fov_mask=None,
                        mode=interp_mode
                    ).detach().cpu().numpy()

                if mod == 'seg':
                    slices = slices.astype(np.uint8)

                nib.save(
                    nib.Nifti1Image(slices, mod_vols[mod]['affine']),
                    path_output.format(folder, 'data_mri', flnm + f"_{mod}.nii.gz")
                )

                slices = slice_pose_torch_batched(
                        vol_t, inv_affine,
                        arr_Ps_torch, arr_d_vecs_torch, arr_i_vecs_torch,
                        dx_mm=0.5, dy_mm=0.5,
                        H_out=192, W_out=192,
                        Y=Y_grid, X=X_grid,
                        P_origin=P_us_torch,
                        fov_mask=fov_slice,
                        mode=interp_mode
                    ).detach().cpu().numpy()

                nib.save(
                    nib.Nifti1Image(slices, mod_vols[mod]['affine']),
                    path_output.format(folder, 'data_mri', flnm + f"_{mod}_fov.nii.gz")
                )
            
            count += 1




