import os
import numpy as np
import random
import nibabel as nib
import time
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from networks.mhvae import MHVAE2D
from utilities.generatesweep import *
from utilities.generate_data import *
from sklearn.decomposition import PCA

LOWER = 0.
UPPER = 99.95

class DatasetReMIND(Dataset):
    def __init__(self, paths_unnorm, paths_norm, mode='training', normalization=False, type_normalization='standardization'):
        assert type_normalization in ['standardization', 'min-max'], \
            f"Normalization {type_normalization} should be in : min-max, standardization"
        self.mode = mode
        self.paths_unnorm = paths_unnorm
        self.paths_norm = paths_norm
        self.normalization = normalization 
        self.type_normalization = type_normalization
        lenghts = [len(k) for k in paths_unnorm.values()]
        assert all(x == lenghts[0] for x in lenghts)
        self.nb_scans = lenghts[0]
        print(mode, self.nb_scans)
        
    def preprocess(self, x, k=[0,0,0], norm=True):
        if self.mode=='training':
            if k[0]==1:
                x = x[::-1, :, :]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[2]==1:
                x = x[:, :, ::-1]
            x = x.copy()
        mask = x>0
        x = x.astype(np.float32)
        if norm:
            if np.any(mask):
                if self.type_normalization == 'standardization':
                    max_data = np.percentile(x, UPPER)
                    x[x>max_data] = max_data
                    sub = np.mean(x)
                    div = 3*np.std(x)
                    x = (x - sub) / (div +1e-8)
                    x[x>1] = 1
                    x[~mask] = -1
                elif self.type_normalization == 'min-max':
                    min_data = np.percentile(x[mask], LOWER)
                    max_data = np.percentile(x[mask], UPPER)

                    x[x>max_data] = max_data
                    x = (x-min_data) / (max_data-min_data)
                    x = x* (1 + 255/256) - 255 / 256
                    
                    x[~mask] = -1.
                    div = (max_data - min_data) / 2
                    sub = min_data
                else:
                    raise NotImplementedError(f"Normalization {self.type_normalization} should be in : min-max, standardization")
            else:
                x -= 1
                div = 1
                sub = 1
        else:
            if np.any(mask):
                x = 2*x - 1
                div = 1/2
                sub = 1/2          
            else:
                x -= 1
                div = 1
                sub = 1
        return x, sub, div


    def __getitem__(self, index):
        output = dict()
        output['fnorm'] = {k:dict() for k in ['sub', 'div']}
        output['fnorm_norm'] = {k:dict() for k in ['sub', 'div']}
        k = [random.randint(0,1), 0, 0]
        for mod in self.paths_unnorm.keys():
            img = nib.load(self.paths_unnorm[mod][index])
            affine = img.affine.squeeze()
            img, sub, div = self.preprocess(img.get_fdata().squeeze(), k, norm=True)
            output[mod] = torch.from_numpy(np.expand_dims(img, axis=0))
            output[mod+'_affine'] = torch.from_numpy(np.expand_dims(affine, axis=0))
            output[mod+'_name'] = os.path.basename(self.paths_unnorm[mod][index].replace('.nii.gz',''))
            output['fnorm']['div'][mod] = div
            output['fnorm']['sub'][mod] = sub
        for mod in self.paths_norm.keys():
            img = nib.load(self.paths_norm[mod][index])
            affine = img.affine.squeeze()
            img, sub, div = self.preprocess(img.get_fdata().squeeze(), k, norm=mod=='us')
            output[mod+'_norm'] = torch.from_numpy(np.expand_dims(img, axis=0))
            output[mod+'_norm_affine'] = torch.from_numpy(np.expand_dims(affine, axis=0))
            output[mod+'_norm_name'] = os.path.basename(self.paths_norm[mod][index].replace('.nii.gz',''))
            output['fnorm_norm']['div'][mod] = div
            output['fnorm_norm']['sub'][mod] = sub            
            
            
        return output

    def __len__(self):
        return self.nb_scans



class DatasetReMINDPred(Dataset):
    def __init__(self, paths_unnorm, mode='training', normalization=False, type_normalization='standardization'):
        assert type_normalization in ['standardization', 'min-max'], \
            f"Normalization {type_normalization} should be in : min-max, standardization"
        self.mode = mode
        self.paths_unnorm = paths_unnorm
        self.normalization = normalization 
        self.type_normalization = type_normalization
        self.nb_cases = len(paths_unnorm)

    def preprocess(self, x, k=[0,0,0], norm=True):
        if self.mode=='training':
            if k[0]==1:
                x = x[::-1, :, :]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[2]==1:
                x = x[:, :, ::-1]
            x = x.copy()
        mask = x>0
        x = x.astype(np.float32)
        if norm:
            if np.any(mask):
                if self.type_normalization == 'standardization':
                    max_data = np.percentile(x, UPPER)
                    x[x>max_data] = max_data
                    sub = np.mean(x)
                    div = 3*np.std(x)
                    x = (x - sub) / (div +1e-8)
                    x[x>1] = 1
                    x[~mask] = -1
                elif self.type_normalization == 'min-max':
                    min_data = np.percentile(x[mask], LOWER)
                    max_data = np.percentile(x[mask], UPPER)

                    x[x>max_data] = max_data
                    x = (x-min_data) / (max_data-min_data)
                    x = x* (1 + 255/256) - 255 / 256
                    
                    x[~mask] = -1.
                    div = (max_data - min_data) / 2
                    sub = min_data
                else:
                    raise NotImplementedError(f"Normalization {self.type_normalization} should be in : min-max, standardization")
            else:
                x -= 1
                div = 1
                sub = 1
        else:
            if np.any(mask):
                x = 2*x - 1
                div = 1/2
                sub = 1/2          
            else:
                x -= 1
                div = 1
                sub = 1
        return x, sub, div


    def __getitem__(self, index):
        output = dict()
        output['fnorm'] = {k:dict() for k in ['sub', 'div']}
        output['fnorm_norm'] = {k:dict() for k in ['sub', 'div']}
        k = [0, 0, 0]
        for mod in self.paths_unnorm[index].keys():
            img = nib.load(self.paths_unnorm[index][mod])
            affine = img.affine.squeeze()
            img, sub, div = self.preprocess(img.get_fdata().squeeze(), k, norm=True)
            output[mod] = torch.from_numpy(np.expand_dims(img, axis=0))
            output[mod+'_affine'] = torch.from_numpy(np.expand_dims(affine, axis=0))
            output[mod+'_name'] = os.path.basename(self.paths_unnorm[index][mod].replace('.nii.gz',''))
            output['fnorm']['div'][mod] = div
            output['fnorm']['sub'][mod] = sub
           
        return output

    def __len__(self):
        return self.nb_cases

class OTFDatasetGenerator(Dataset):
    def __init__(self, total_subsets_mr=None, ultrasounds=None, filtered_arr=None, weights=None, m_2=None, tumor_points_3d=None, surface_tree=None, radius_mm=None, NB_FRAMES=1, max_dist=1, video_p=0):

        self.total_subsets_mr = total_subsets_mr
        self.ultrasounds = ultrasounds
        self.filtered_arr = filtered_arr
        self.weights = weights
        self.m_2 = m_2
        self.TUMOR_POINTS_3D = tumor_points_3d
        self.surface_tree = surface_tree
        self.radius_mm = radius_mm
        self.NB_FRAMES = NB_FRAMES
        self.max_dist = max_dist
        self.video_p = video_p

    def __len__(self):
        return 1000 
    
    def __getitem__(self, index):

        us = next(self.ultrasounds)
        Ps, d_vecs, i_vecs, P_origin, fov_slice = self.generate_us_sweep(us)
        sample = {'Ps': Ps, 'd_vecs': d_vecs, 'i_vecs': i_vecs, 'P_origin': P_origin, 'fov_slice': fov_slice}
        
        return sample

    def generate_us_sweep(self, us):
        max_attempts = 10
        for attempt in range(max_attempts):
            chosen_idx = np.random.choice(self.filtered_arr.shape[0], p=self.weights)
            P_0 = self.filtered_arr[chosen_idx]

            # Load US + probe mask + keypoints
            us_sitk = sitk.ReadImage(f'../data/coregistered/us-space/{us}/{us}-us.nii.gz')
            kp_img = nib.load(f'../data/precomputed_us_masks/{us}/{us}-keypoints.nii.gz')
            keypoints = kp_img.get_fdata()
            frames_w_surface = np.argwhere(keypoints==4)[:, 2]

            fov_img = nib.load(f'../data/precomputed_us_masks/{us}/{us}-fov_mask.nii.gz')
            fov_mask = fov_img.get_fdata()
            fov_mask = torch.from_numpy(fov_mask).to(torch.float32)#.to(self.device)

            z_mid = np.random.choice(frames_w_surface)

            fov_slice = fov_mask[...,z_mid]
            points_mask = keypoints[...,z_mid]
            ys, xs = np.where(points_mask > 0)
            labels = points_mask[ys, xs]

            idx = labels == 4
            P_us = [xs[idx][0], ys[idx][0]]
            P_us = torch.from_numpy(np.array(P_us)).to(torch.float32)#.to(self.device)

            # --------------------
            # 5) Determine how to scan
            # --------------------
            # Define d_vec_0 pointing from P_0 to m_2
            d_vec_0 = self.m_2 - P_0
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

            for T in self.TUMOR_POINTS_3D:
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
            T_prime = self.TUMOR_POINTS_3D - P_0

            TUMOR_POINTS_1D = np.dot(T_prime, sweep_dir)

            tumor_min = np.min(TUMOR_POINTS_1D)
            tumor_max = np.max(TUMOR_POINTS_1D)

            tumor_min_3D = P_0 + (tumor_min * sweep_dir) 
            tumor_max_3D = P_0 + (tumor_max * sweep_dir)

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

            # Sweep along sweep_dir updating Ps and keeping the image plane constant (the number of sweep frames is constant = 90)
            sample_Ps, Ps_before, Ps_after = sample_points_along_straight_line(P_0, sweep_dir, tumor_min, tumor_max)

            Ps_after = walk_on_surface(
                self.filtered_arr,
                x0_world=P_0,
                n_world=sweep_dir,
                dx_mm=0.5,
                n_steps=len(Ps_after),
                len_max_mm=np.abs(tumor_max),
                radius_mm=self.radius_mm,
            )

            Ps_before = walk_on_surface(
                self.filtered_arr,
                x0_world=P_0,
                n_world=-sweep_dir,
                dx_mm=0.5,
                n_steps=len(Ps_before),
                len_max_mm=np.abs(tumor_min),
                radius_mm=self.radius_mm,
            )

            # Check if result is empty
            if Ps_after.shape[0] > 0 and Ps_before.shape[0] > 0:
                break  
            elif attempt == max_attempts - 1:
                raise RuntimeError("walk_on_surface_bidirectional returned empty result after multiple attempts.")

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

        arr_Ps_torch = torch.as_tensor(arr_Ps).to(torch.float32)
        arr_d_vecs_torch = torch.as_tensor(arr_d_vecs).to(torch.float32)
        arr_i_vecs_torch = torch.as_tensor(arr_i_vecs).to(torch.float32)

        return arr_Ps_torch, arr_d_vecs_torch, arr_i_vecs_torch, P_us, fov_slice