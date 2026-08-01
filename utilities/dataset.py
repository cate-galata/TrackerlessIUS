import os
import numpy as np
import random
import nibabel as nib
import time
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

LOWER = 0.
UPPER = 99.95

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
    """
    On-the-fly sweep sampler used during training.

    Each item returns only:
      - sweep positions/orientations
      - US reference anchor
      - FOV mask slice

    The actual MRI resampling and synthesis still happen in the training loop.
    """
    def __init__(
        self,
        sweep_generator,
        context,
        ultrasounds,
        device=None,
        length: int = 1000,
    ):
        self.sweep_generator = sweep_generator
        self.context = context
        self.ultrasounds = ultrasounds
        self.device = device
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        us_case = next(self.ultrasounds)

        ps, d_vecs, i_vecs, p_us, fov_slice = self.sweep_generator.sample_sweep(
            context=self.context,
            us_case=us_case,
            device=torch.device("cpu"),
        )

        return {
            "Ps": ps,
            "d_vecs": d_vecs,
            "i_vecs": i_vecs,
            "P_origin": p_us,
            "fov_slice": fov_slice,
        }