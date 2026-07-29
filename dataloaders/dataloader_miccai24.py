
from glob import glob

import monai
import numpy as np
import torch
from monai.data import DataLoader
from monai.transforms import (
    RandAffined, 
    LoadImaged, 
    EnsureChannelFirstd, 
    NormalizeIntensityd, 
    SampleForegroundLocationsd, 
    RandGaussianNoised, 
    RandGaussianSmoothd, 
    RandScaleIntensityd, 
    RandScaleIntensityFixedMeand, 
    RandSimulateLowResolutiond, 
    RandAdjustContrastd, 
    RandFlipd, 
    Compose, 
    Identity,
    EnsureTyped, 
)

modality_keys = ["img"]
label_keys = ["seg"]
all_keys = modality_keys + label_keys

patch_size= [128, 128, 128]
pos_sample_num = 2
neg_sample_num = 1
use_nonzero = True
use_prior = False
    
def get_training_loader(train_files, batch_size, num_workers, use_spacial_augmentation=False, cache=False, seed=0):

    load_image = LoadImaged(keys=all_keys, image_only=True)
    ensure_channel_first = EnsureChannelFirstd(keys=all_keys)
    norm_transform = NormalizeIntensityd(keys=modality_keys, nonzero=use_nonzero)
    
    if use_spacial_augmentation:
    
        sample_foreground_locations = SampleForegroundLocationsd(label_keys=label_keys, num_samples=10000)

        rand_affine = RandAffined(
            keys=all_keys,
            mode=(3,)*len(modality_keys) + ("nearest", ),  # 3 means third order spline interpolation
            prob=1.0,
            rotate_range= (30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi),
            prob_rotate=0.2,
            translate_range=(0, 0, 0),
            foreground_oversampling_prob=pos_sample_num / neg_sample_num,
            label_key_for_foreground_oversampling="seg",
            prob_translate=1.0,
            scale_range=((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)),
            prob_scale=0.2,
            padding_mode=("constant",)*len(modality_keys) + ("border", ),
        )
    else:
        sample_foreground_locations = Identity()
        rand_affine = Identity()

    rand_gauss_noise = RandGaussianNoised(keys=modality_keys, std=0.1, prob=0.2)

    rand_gauss_smooth = RandGaussianSmoothd(keys=modality_keys,
                                            sigma_x=(0.5, 1.0),
                                            sigma_y=(0.5, 1.0),
                                            sigma_z=(0.5, 1.0),
                                            prob=0.4, )  # 0.5 comes from the per_channel_probability

    scale_intensity = RandScaleIntensityd(keys=modality_keys, factors=[-0.25, 0.25], prob=0.30)

    shift_intensity = RandScaleIntensityFixedMeand(keys=modality_keys, factors=[-0.25, 0.25], preserve_range=True,
                                                prob=0.30)

    sim_lowres = RandSimulateLowResolutiond(keys=modality_keys, prob=0.45, zoom_range=(0.5, 1.0))

    adjust_contrast_inverted = RandAdjustContrastd(keys=modality_keys, prob=0.1 * 1.0, gamma=(0.7, 1.5),
                                                invert_image=True, retain_stats=True)

    adjust_contrast = RandAdjustContrastd(keys=modality_keys, prob=0.6 * 1.0, gamma=(0.7, 1.5), invert_image=False,
                                        retain_stats=True)

    mirror_x = RandFlipd(all_keys, spatial_axis=[0], prob=0.5)
    mirror_y = RandFlipd(all_keys, spatial_axis=[1], prob=0.5)
    mirror_z = RandFlipd(all_keys, spatial_axis=[2], prob=0.5)

    train_transforms = Compose(
        [
            load_image,
            ensure_channel_first,
            norm_transform,  # -1
            sample_foreground_locations,  # 0
            rand_affine,  # 1
            rand_gauss_noise,  # 2
            rand_gauss_smooth,  # 3
            scale_intensity,  # 4
            shift_intensity,  # 5
            sim_lowres,  # 6
            adjust_contrast_inverted,  # 7
            adjust_contrast,  # 8
            mirror_x, mirror_y, mirror_z,  # 9
            EnsureTyped(keys=modality_keys),
        ], 
        unpack_items=True)

    train_transforms.set_random_state(seed=seed) 
    # 3D dataset with preprocessing transforms
    if not cache:
        volume_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    else:
        volume_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
    # use batch_size=1 to check the volumes because the input volumes have different shapes
    train_loader = DataLoader(volume_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=num_workers > 0, generator=torch.Generator().manual_seed(seed))
    check_data = monai.utils.misc.first(train_loader)
    print("training volume's shape: ", check_data["img"].shape, check_data["seg"].shape)
    return train_loader


def get_validation_loader(val_files, batch_size, num_workers, cache=False):
    # volume-level transforms for both image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            NormalizeIntensityd(keys=modality_keys, nonzero=use_nonzero),
            # ScaleIntensityd(keys="img"),
            # RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    # 3D dataset with preprocessing transforms
    if not cache:
        volume_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    else:
        volume_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)

    # use batch_size=1 to check the volumes because the input volumes have different shapes
    val_loader = DataLoader(volume_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,)
    check_data = monai.utils.misc.first(val_loader)
    print("validation volume's shape: ", check_data["img"].shape, check_data["seg"].shape)  
    return val_loader