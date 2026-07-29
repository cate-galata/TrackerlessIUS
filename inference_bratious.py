#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm

import numpy as np
import nibabel as nib

import torch
from torch import nn
from torch.utils.data import DataLoader

import monai
from monai.inferers import SliceInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
)
from monai.utils import set_determinism

from networks.unet import UNet2D
import time

import torch.nn.functional as F

def resize_volume_2d(x, size, mode="bilinear"):
    """
    Resize the spatial (H, W) dims of a tensor shaped (B, C, H, W, D),
    keeping batch, channel, and depth (slice) dims unchanged.
    """
    b, c, h, w, d = x.shape
    # move D into batch so interpolate only touches H, W
    x = x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
    align_corners = False if mode in ("bilinear", "bicubic") else None
    x = F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    x = x.reshape(b, d, c, *size).permute(0, 2, 3, 4, 1)
    return x


PHASES = ["validation"]
NB_CLASSES = 2


def inference(paths_dict, model, fold_dir_model, device, save_path, opt):

    print("[INFO] Starting inference")
    print(f"Case {opt.case}")

    val_transform = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=True),
            EnsureTyped(keys=["img"]),
        ]
    )
    val_ds = monai.data.Dataset(data=paths_dict["validation"], transform=val_transform)
    data_loader = DataLoader(val_ds, num_workers=1, pin_memory=torch.cuda.is_available())

    # Load checkpoint 
    checkpoint_path = os.path.join(fold_dir_model, "./CP_{}.pth").format(opt.epoch_inf)
    assert os.path.isfile(checkpoint_path), f"no checkpoint found {checkpoint_path}"
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    # UNet2D returns a list/tuple; original code uses the first output
    def model_seg(*args, **kwargs):
        return model(*args, **kwargs)[0]

    # Slice inferer config 
    roi_size = (512, 512)
    sw_batch_size = 128
    slice_inferer = SliceInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        spatial_dim=2,
        device=torch.device("cpu"),
        padding_mode="replicate",
    )

    # Inference loop
    for batch in tqdm(data_loader):
        inputs = batch["img"].to(device)

        affine = batch["img"].meta["original_affine"]
        name = os.path.basename(batch["img"].meta["filename_or_obj"])

        orig_size = inputs.shape[2:4]

        with torch.no_grad():

            inputs_resized = resize_volume_2d(inputs, size=(512, 512), mode="bilinear")

            pred = slice_inferer(inputs_resized, model_seg).softmax(1)
            pred = resize_volume_2d(pred, size=orig_size, mode="bilinear").numpy()
            pred = np.argmax(pred, axis=1).astype(np.uint8)[0]

            nib.save(nib.Nifti1Image(pred, affine), os.path.join(save_path, name))


def main():
    opt = parsing_data()
    set_determinism(seed=opt.seed)

    # Folders
    fold_dir = os.path.join(opt.model_dir, "bratious", str(opt.learning_rate), str(opt.spacial), opt.comment)
    fold_dir_model = os.path.join(fold_dir, "models")

    save_path = os.path.join(fold_dir, f"inference_{opt.case}_{opt.epoch_inf}")
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Validation "dataset" with a single image
    assert opt.path_data_true is not None, "--path_data_true is required"
    assert os.path.exists(opt.path_data_true), f"error gt img {opt.path_data_true}"

    paths_dict = {split: [] for split in PHASES}
    paths_dict["validation"].append({"img": opt.path_data_true})

    # Model
    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

    model = UNet2D(
        input_channels=1,
        base_num_features=32,
        num_classes=NB_CLASSES,
        num_pool=5,
        conv_op=nn.Conv2d,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs=norm_op_kwargs,
        nonlin=net_nonlin,
        nonlin_kwargs=net_nonlin_kwargs,
    ).to(device)


    inference(paths_dict, model, fold_dir_model, device, save_path, opt)


def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using extreme points as supervision"
    )

    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of the batch size (default: 64)")
    parser.add_argument("--case", type=str, default="Case112")

    parser.add_argument("--path_data_true", type=str, default=None, help="Path to the input image")
    
    parser.add_argument("--comment", type=str, default="", help="Experiment comment/tag")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate")

    parser.add_argument("--epoch_inf", type=str, default="final")
    parser.add_argument("--spacial", action="store_true", help="Aug spacial abled")

    parser.add_argument('--seed', type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    main()
