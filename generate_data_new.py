import SimpleITK as sitk
import numpy as np
import pandas as pd
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
from utilities.utils import set_determinism

ALL_CASES = ["Case011", "Case025", "Case027", "Case045", "Case052", "Case056", "Case070", "Case074", "Case085", "Case099", "Case103", "Case112", "Case114"]

parser = argparse.ArgumentParser(
                    prog='Generatedata',
                    description='This script creates ultrasound sweeps on MRI data.')

parser.add_argument('--K', type=int, default=1, help='Number of K sweeps')
parser.add_argument('--annotator', type=str, default="remind", help='Annotator')
parser.add_argument('--case', type=str, default="Case027", help='Case')
parser.add_argument('--seed', type=int, default=0, help='Seed')


args = parser.parse_args()

NB_cases = args.K
annotator = args.annotator
case = args.case
assert case in ALL_CASES, f"Error {case} in not in {ALL_CASES}"
seed = args.seed


path_data = f'./data/coregistered/mri-space'
reg_folder = f'./data/registration/mni'
us_path_folder = f'./data/coregistered/us-space'
path_strip = "./experiments/skullstripping_hdbet/"
mni_surf_path = './data/mni/mni_brain_surface.nii.gz.seg.nrrd'
mni_non_viable_path = './data/mni/refined_non_viable_surface_mask2.nii.gz'

path_output = "./experiments/synthetic/{}-{}-{}/{}/{}/{}"
path_label = f"./experiments/synthetic/label-{annotator}/label-mr-space/{case}/seg.nii.gz"


set_determinism(seed=seed)


# --------------------
# 1) Reference MR and segmentation mask
# --------------------
# MR modalities + all non-empty subsets
images_modalities = get_coregistered_mr_images(path_data, case)
modalities = [k[0] for k in images_modalities]
total_subsets_mr = list(
    chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1))
)

print(total_subsets_mr)

path_mri = glob.glob(os.path.join(path_data, case, f'{case}-**_ref.nii.gz'))[0]
img_mri = sitk.ReadImage(path_mri)
img_nib = nib.load(path_mri)
mri_data = img_nib.get_fdata()       
affine = torch.from_numpy(img_nib.affine)

mod_vols = {}
for mod in modalities:
    path = glob.glob(os.path.join(path_data, case, f'{case}-{mod}**.nii.gz'))[0]
    img = nib.load(path)
    data = img.get_fdata() 
    inv_affine = torch.linalg.inv(torch.from_numpy(img.affine))
    vol_np = np.transpose(data, (2, 1, 0))
    vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)
    mod_vols[mod] = {
        'volume': vol_t,
        'affine': img.affine,
        'inv_affine': inv_affine
    }

path_target = os.path.join(path_data, case, f'{case}-target.nii.gz')
img_target = sitk.ReadImage(path_target)
target_nib = nib.load(path_target)
data_target = target_nib.get_fdata()        
target_affine = torch.from_numpy(target_nib.affine)

vol_target_np = np.transpose(data_target, (2, 1, 0))
vol_target_t = torch.from_numpy(vol_target_np).unsqueeze(0).unsqueeze(0)
device = vol_target_t.device
inv_affine_target = torch.linalg.inv(target_affine).to(device)

# --------------------
# 2) Prepare output folders + candidate US list
# --------------------
for folder in ["data", "data_mri", "landmarks", "reg", "label", "tracking"]:
    os.makedirs(path_output.format(annotator, NB_cases, seed, case, folder, ""), exist_ok=True)

ultrasounds = [k for k in os.listdir(us_path_folder) if "Case" in k and case not in k]

# --------------------
# 3) Determine where to scan
# --------------------
path_img_strip = glob.glob(os.path.join(path_strip, case, f"{case}-**_mask.nii.gz"))[0]
data_noncerebrum = (nib.load(path_img_strip).get_fdata() == 0)
border = find_boundaries(data_noncerebrum, mode="inner")

border_pixels = np.stack(np.where(border),-1)

new_arr = vox_to_world_many(nib.load(path_label).affine, border_pixels)


# Register viable craniotomy locations from the labeled MNI template
affine_path = os.path.join(reg_folder, case, f"{case}-mri-to-mni-Syn0GenericAffine.mat")
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

radius_mm = np.sqrt(img_mri.GetSpacing()[0]**2 + img_mri.GetSpacing()[1]**2 + img_mri.GetSpacing()[2]**2)

idx = tree.query_ball_point(new_arr, r=radius_mm) # for every point in new_arr, idx returns the points in the tree which are within distance r
mask = np.array([len(n) == 0 for n in idx])
filtered_arr = new_arr[mask]

surface_tree = cKDTree(filtered_arr) if len(filtered_arr) > 0 else None

tumor_voxels = np.argwhere(data_target > 0)
TUMOR_POINTS_3D = vox_to_world_many(target_nib.affine, tumor_voxels)

m_2 = [k.mean() for k in np.where(data_target > 0)]
m_2 += np.random.normal(loc=0.0, scale=3.0, size=(3))
m_2 = target_nib.affine.dot(m_2.tolist()+[1])[:3]


d_2 = np.linalg.norm(filtered_arr - m_2, axis=1)
weights = np.exp(-d_2)
weights /= weights.sum()

# --------------------
# 4) For each modality subset: generate NB_cases synthetic pairs (MR->US)
# --------------------
for modalities_set in total_subsets_mr:

    ultrasounds_subset = iter(np.random.choice(ultrasounds, NB_cases, replace=False).tolist())

    count = 0
    while count < NB_cases:
        us = next(ultrasounds_subset)
        chosen_idx = np.random.choice(filtered_arr.shape[0], p=weights)
        P_0 = filtered_arr[chosen_idx]

        # Load US + probe mask + keypoints
        kp_img = nib.load(f'./data/precomputed_us_masks/{us}/{us}-keypoints.nii.gz')
        keypoints = kp_img.get_fdata()
        frames_w_surface = np.argwhere(keypoints==4)[:, 2]
        N = len(frames_w_surface)
        period = 2 * (N - 1) if N > 1 else 1  

        fov_img = nib.load(f'./data/precomputed_us_masks/{us}/{us}-fov_mask.nii.gz')
        fov_mask = fov_img.get_fdata()
        fov_mask = torch.from_numpy(fov_mask).to(device)

        z_mid = np.random.choice(frames_w_surface)

        points_mask = keypoints[..., z_mid]
        fov_slice = fov_mask[..., z_mid]
        ys, xs = np.where(points_mask > 0)
        labels = points_mask[ys, xs]

        idx = labels == 4
        P_us = np.array([xs[idx][0], ys[idx][0]])  
        P_us = torch.from_numpy(P_us).to(device)

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

        # Find min and max coordinates of tumor along sweep_dir
        T_prime = TUMOR_POINTS_3D - P_0

        TUMOR_POINTS_1D = np.dot(T_prime, sweep_dir)

        tumor_min = np.min(TUMOR_POINTS_1D)
        tumor_max = np.max(TUMOR_POINTS_1D)

        tumor_min_3D_eps = P_0 + (tumor_min * sweep_dir) 
        tumor_max_3D_eps = P_0 + (tumor_max * sweep_dir)
        idx_min = np.argmin(TUMOR_POINTS_1D)
        idx_max = np.argmax(TUMOR_POINTS_1D)
        tumor_min_3D = TUMOR_POINTS_3D[idx_min]
        tumor_max_3D = TUMOR_POINTS_3D[idx_max]

        # Define i_vec_0 orthogonal to d_vec_0 and sweep_dir
        i_vec_0 = np.cross(d_vec_0, sweep_dir)
        i_vec_0 = i_vec_0 / np.linalg.norm(i_vec_0)

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

        # Save points on sweep trajectory and image plane vectors to a dataframe -> each row represents one frame
        arr_Ps = np.array(Ps)
        arr_d_vecs = np.array(d_vecs)
        arr_i_vecs = np.array(i_vecs)

        arr_Ps_torch = torch.as_tensor(arr_Ps)
        arr_d_vecs_torch = torch.as_tensor(arr_d_vecs)
        arr_i_vecs_torch = torch.as_tensor(arr_i_vecs)

        listToStr = "-".join([str(elem) for elem in modalities_set])
        flnm = f"{case}-{us}-{listToStr}"

        # Resample MR modalities into US space
        for mod in modalities_set:
            vol_t = mod_vols[mod]['volume'].to(device)
            inv_affine = mod_vols[mod]['inv_affine'].to(device)
            device = vol_t.device

            volume_slices = []
            for idx in range(arr_Ps.shape[0]):
                P = arr_Ps_torch[idx].to(device)
                d_vec = arr_d_vecs_torch[idx].to(device)
                i_vec = arr_i_vecs_torch[idx].to(device)

                slice_2d_torch, slice_coords = slice_pose_torch(
                    vol_t,
                    inv_affine,
                    point_world=P,
                    dir_world=d_vec,
                    inplane_world=i_vec,
                    dx_mm=0.5,
                    dy_mm=0.5,
                    H_out=192,
                    W_out=192,
                    P_origin=P_us,
                    fov_mask=fov_slice,
                    pad_value=0.0,
                    return_world_coords=True,
                    mode='bilinear'
                )

                volume_slices.append(slice_2d_torch.detach().cpu().numpy())

            volume_3d = np.stack(volume_slices, axis=-1)

            nib.save(
                nib.Nifti1Image(volume_3d, mod_vols[mod]['affine']),
                path_output.format(annotator, NB_cases, seed, case, 'data_mri', flnm + f"_{mod}.nii.gz")
            )

        # Resample segmentation into US space
        volume_slices = []
        for idx in range(arr_Ps.shape[0]):
            P = arr_Ps_torch[idx].to(device)
            d_vec = arr_d_vecs_torch[idx].to(device)
            i_vec = arr_i_vecs_torch[idx].to(device)

            slice_2d_torch, slice_coords = slice_pose_torch(
                vol_target_t,
                inv_affine_target,
                point_world=P,
                dir_world=d_vec,
                inplane_world=i_vec,
                dx_mm=0.5,
                dy_mm=0.5,
                H_out=192,
                W_out=192,
                P_origin=P_us,
                fov_mask=fov_slice,
                pad_value=0.0,
                return_world_coords=True,
                mode='nearest'
            )

            volume_slices.append(slice_2d_torch.detach().cpu().numpy().astype(np.uint8))

        volume_3d = np.stack(volume_slices, axis=-1)

        nib.save(
            nib.Nifti1Image(volume_3d, target_nib.affine),
            path_output.format(annotator, NB_cases, seed, case, 'data_mri', flnm + f"_seg.nii.gz")
        )

        count += 1







