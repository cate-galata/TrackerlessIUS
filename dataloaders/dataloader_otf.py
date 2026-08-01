
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

from utilities.dataset import OTFDatasetGenerator

modality_keys = ["img"]
label_keys = ["seg"]

use_nonzero = True
    
def get_training_loader(
    batch_size,
    num_workers,
    sweep_generator,
    context,
    ultrasounds,
    seed=0,
):
    volume_ds = OTFDatasetGenerator(
        sweep_generator=sweep_generator,
        context=context,
        ultrasounds=ultrasounds,
        length=1000,
    )

    train_loader = DataLoader(
        volume_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return train_loader


def get_validation_loader(val_files, batch_size, num_workers, cache=False):
    # volume-level transforms for both image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"], channel_dim='no_channel'),
            NormalizeIntensityd(keys=modality_keys, nonzero=use_nonzero),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    # 3D dataset with preprocessing transforms
    if not cache:
        volume_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    else:
        volume_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)

    # use batch_size=1 to check the volumes because the input volumes have different shapes
    val_loader = DataLoader(volume_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    check_data = monai.utils.misc.first(val_loader)
    print("first validation volume's shape: ", check_data["img"].shape, check_data["seg"].shape)  

    return val_loader


def get_test_loader(test_files, batch_size, num_workers, cache=False):
    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"], channel_dim='no_channel'),
            NormalizeIntensityd(keys=modality_keys, nonzero=use_nonzero),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    # 3D dataset with preprocessing transforms
    if not cache:
        volume_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    else:
        volume_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms)

    # use batch_size=1 to check the volumes because the input volumes have different shapes
    test_loader = DataLoader(volume_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    check_data = monai.utils.misc.first(test_loader)
    print("first test volume's shape: ", check_data["img"].shape, check_data["seg"].shape)  

    return test_loader

    
