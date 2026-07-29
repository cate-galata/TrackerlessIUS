import argparse
import os
import pickle
import time
import sys
import itertools
import torch_tensorrt
import monai

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm
from monai.utils import set_determinism
from itertools import chain, combinations

from networks.unet import UNet2D
from networks.mhvae import MHVAE2D
from utilities.dataloader_miccai24_new import *
from utilities.losses import *
from utilities.utils import create_logger, infinite_iterable, poly_lr, draw_curve
from utilities.generation import *
from utilities.generatesweep import *
from utilities.generate_data import generate_us_sweep, synthesize_us_sweep

import monai.transforms.compose
import monai.transforms.transform
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation

# ----------------------------
# Constants / defaults
# ----------------------------
PHASES = ["training", "validation", "test"]
NB_CLASSES = 2
CACHE = False

WEIGHT_DECAY = 3e-5
WORKERS = 0

modality_keys = ["img"]
label_keys = ["seg"]
all_keys = modality_keys + label_keys

patch_size= [128, 128, 128]
pos_sample_num = 2
neg_sample_num = 1
use_nonzero = True
use_prior = False

H_out = 192
W_out = 192

SAFE_MAX = np.iinfo(np.uint32).max 
monai.transforms.compose.MAX_SEED = SAFE_MAX
monai.transforms.transform.MAX_SEED = SAFE_MAX

def visualize_batch(batch, save_path, nrow=4):
    """
    Saves a grid of images and their corresponding masks.
    Assumes batch['img'] and batch['seg'] are tensors.
    """
    # Take the first few samples from the batch
    imgs = batch['img'].as_tensor().cpu()
    segs = batch['seg'].as_tensor().cpu().float()
    
    # Normalize images to [0, 1] if they aren't already for visualization
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
    
    # Create a grid: Top row Images, Bottom row Masks
    grid = torchvision.utils.make_grid(torch.cat([imgs, segs], dim=0), nrow=nrow)
    
    torchvision.utils.save_image(grid, save_path)
    

def get_training_augmentation(device, use_spacial_augmentation=True, seed=0):
    ensure_channel_first = EnsureChannelFirstd(keys=all_keys, channel_dim='no_channel')
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
            padding_mode=("constant",)*len(modality_keys) + ("constant", ),
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
            EnsureTyped(keys=modality_keys, device=device),
        ], 
        unpack_items=True)

    train_transforms.set_random_state(seed=seed) 
    
    return train_transforms


def train(paths_dict, synthesizer, model, criterion, soft_criterion, criterion_ce, soft_ce, device, save_path, logger, opt, mod_vols=None, total_subsets_mr=None, ultrasounds=None, filtered_arr=None, weights=None, m_2=None, tumor_points_3d=None, surface_tree=None, radius_mm=None):
    logger.info("[INFO] Starting training")
    logger.info(f"Case {opt.case}")
    torch.cuda.synchronize()
    since = time.perf_counter()

    const_iter = len([k for k in os.listdir(f"./miccai2024_data/synthetic/remind-10-2/{opt.case}/synthetic_data_reslice")]) - 20
    print('const_iter ', const_iter)

    # Dataloaders
    ind_batch = {
        "training": np.arange(0, min(1000, const_iter), opt.batch_size),
        "validation": np.arange(0, min(1000, len(paths_dict["validation"])), 1),
        "test": np.arange(0, len(paths_dict["test"]), 1)
    }

    dataloaders = {
        "training": infinite_iterable(
            get_training_loader(
                mod_vols=mod_vols, 
                total_subsets_mr=total_subsets_mr, 
                ultrasounds=ultrasounds, 
                filtered_arr=filtered_arr, 
                weights=weights, 
                m_2=m_2, 
                tumor_points_3d=tumor_points_3d, 
                surface_tree=surface_tree, 
                batch_size=opt.batch_size,
                num_workers=WORKERS,
                device=device,
                case=opt.case,
                radius_mm=radius_mm,
                use_spacial_augmentation=opt.spacial,
                seed=opt.seed,
            )
        ),
        "validation": infinite_iterable(
            get_validation_loader(
                paths_dict["validation"],
                batch_size=1,
                num_workers=0,
                cache=CACHE,
            )
        ),
        "test": infinite_iterable(
            get_test_loader(
                paths_dict["test"],
                batch_size=1,
                num_workers=0,
                cache=CACHE,
            )
        )
    }

    augment = get_training_augmentation(device, use_spacial_augmentation=opt.spacial, seed=opt.seed)
    ys = torch.arange(H_out, device=device, dtype=torch.float32)
    xs = torch.arange(W_out, device=device, dtype=torch.float32)

    Y_grid, X_grid = torch.meshgrid(ys, xs, indexing="ij")

    # Experiment folder / logging
    folder_case = os.path.join(opt.model_dir, opt.case, str(opt.learning_rate), str(opt.spacial), opt.comment)
    df_path = os.path.join(folder_case, "log.csv")
    df_path_test = os.path.join(folder_case, "log_test.csv")
    df_path_train = os.path.join(folder_case, "log_train.csv")

    df = pd.DataFrame(columns=["case", "epoch", "wall_time", "val_dice", "best_epoch", "best_val", "lr"])
    df_test = pd.DataFrame(columns=["case", "epoch", "wall_time", "test_dice"])
    df_train = pd.DataFrame(columns=["case", "epoch", "wall_time", "train_dice"])
    best_val = None
    best_epoch = 0
    epoch = 0

    initial_lr = opt.learning_rate
    base_lr = opt.learning_rate

    # Optimizer (nnU-Net-like)
    optimizer = torch.optim.SGD(
        model.parameters(),
        initial_lr,
        weight_decay=WEIGHT_DECAY,
        momentum=0.99,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler()

    infos = {
        "training": [],
        "validation": [],
        "test": [],
        "epochs": [],
        "path": os.path.join(folder_case, "train.jpg"),
    }

    data_time_avg = []
    train_time_avg = []
    data_time_gen_avg = []
    data_time_syn_avg = []
    data_time_aug_avg = []

    # Training loop
    continue_training = True
    while continue_training:
        epoch += 1
        logger.info("-" * 10)
        logger.info(f"Epoch {epoch}/")

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Current learning rate is: {current_lr:.4f}")

        draw_curve(infos)
        infos["epochs"].append(epoch)

        for phase in PHASES[::-1]:
            is_train = phase == "training"
            model.train() if is_train else model.eval()

            running_loss = 0.0
            epoch_samples = 0

            for _ in tqdm(ind_batch[phase]):

                torch.cuda.synchronize()
                data_time = time.perf_counter()
                batch = next(dataloaders[phase])

                if is_train:
                    torch.cuda.synchronize()
                    data_train_generation = time.perf_counter()
                    Ps = batch['Ps'].to(device)
                    d_vecs = batch['d_vecs'].to(device)
                    i_vecs = batch['i_vecs'].to(device)
                    P_origin = batch['P_origin'].to(device)
                    fov_slice = batch['fov_slice'].to(device)

                    output = dict()
                    modalities_set = random.choice(total_subsets_mr)
                    # for mod in modalities_set+['target']:
                    for mod in modalities_set+('target',):
                        vol_t = mod_vols[mod]['volume'].to(device)
                        inv_affine = mod_vols[mod]['inv_affine'].to(device)
                        interp_mode = "nearest" if mod == "target" else "bilinear" 

                        slices = slice_pose_torch_shared_volume_batched(
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
                            P_origin=P_origin,
                            fov_mask=fov_slice,
                            mode=interp_mode
                        ) # (N, H, W)

                        output[mod] = slices.permute(0,2,3,1) # (B, H, W, N)

                    torch.cuda.synchronize()
                    data_train_generation = time.perf_counter() - data_train_generation
                    data_time_gen_avg.append(data_train_generation)
                    torch.cuda.synchronize()
                    data_train_synthesis = time.perf_counter()
                    pred = synthesize_us_sweep(synthesizer, modalities_set, output, device, opt.type_normalization)
                    pred = (pred + 1) / 2  # [-1, 1] -> [0, 1]
                    torch.cuda.synchronize()
                    data_train_synthesis = time.perf_counter() - data_train_synthesis
                    data_time_syn_avg.append(data_train_synthesis)

                    image = pred.squeeze(1)
                    image = image.permute(1, 2, 0)  # (H,W,D)

                    seg_data = output['target'].squeeze(0) # (H,W,D)
                    
                    batch = {'img': image, 'seg': seg_data}

                    torch.cuda.synchronize()
                    data_train_augmentation = time.perf_counter()
                    batch = augment(batch)
                    torch.cuda.synchronize()
                    data_train_augmentation = time.perf_counter() - data_train_augmentation
                    data_time_aug_avg.append(data_train_augmentation)

                    for mod in ["img", "seg"]:
                        batch[mod] = batch[mod].unsqueeze(0)

                # Match the original tensor reshaping
                for mod in ["img", "seg"]:
                    batch[mod] = (
                        batch[mod]
                        .permute(0, 4, 1, 2, 3)
                        .reshape(-1, 1, *batch[mod].shape[2:4])
                        .to(device)
                    )

                inputs = batch["img"]
                labels = (batch["seg"] > 0).float()

                torch.cuda.synchronize()
                data_time = time.perf_counter() - data_time
                data_time_avg.append(data_time)

                if epoch == 1:
                    visualize_batch(batch, os.path.join(folder_case, f"debug_batch_{phase}.png"))

                    if is_train:
                        print(f'Epoch 0-1 average time - simulation: {np.mean(data_time_gen_avg):.3f}s synthesis: {np.mean(data_time_syn_avg):.3f}s augmentation: {np.mean(data_time_aug_avg):.3f}s')

                torch.cuda.synchronize()
                train_time = time.perf_counter()
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    with torch.cuda.amp.autocast():        
                        outputs = model(inputs)

                        if phase == "validation" or phase == "test":
                            output = torch.argmax(outputs[0], 1, keepdim=True)
                            dice = (2 * (output * labels).sum()) / (output.sum() + labels.sum() + 10e-20)
                            loss = -dice
                        else:
                            loss = criterion(outputs[0], labels)
                            for i, o in enumerate(outputs):
                                loss += soft_criterion(o, labels) / (2 ** (i + 1))

                    if is_train:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += 1
                running_loss += loss.item()

                torch.cuda.synchronize()
                train_time = time.perf_counter() - train_time
                train_time_avg.append(train_time)

                if epoch % 10 == 0:
                    print(f"Epoch 0-{epoch} average time - data loading: {np.mean(data_time_avg):.3f}s  training: {np.mean(train_time_avg):.3f}s")

            epoch_loss = running_loss / epoch_samples if epoch_samples else 0.0
            logger.info(f"{phase}  Loss : {epoch_loss:.4f}")

            if phase == "validation" or phase == "test":
                infos[phase].append(-epoch_loss)
            else:
                infos[phase].append(epoch_loss)

            # Validation bookkeeping
            if phase == "validation":
                if best_val is None or epoch_loss <= best_val:
                    best_val = epoch_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format("best"))

                if epoch_loss < -0.90:
                    base_lr = opt.learning_rate / 10

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [
                                {
                                    "case": opt.case,
                                    "epoch": epoch,
                                    "wall_time": time.time() - since,
                                    "val_dice": -epoch_loss,
                                    "best_epoch": best_epoch,
                                    "best_val": best_val,
                                    "lr": optimizer.param_groups[0]["lr"],
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                df.to_csv(df_path, index=False)

                optimizer.param_groups[0]["lr"] = poly_lr(epoch, opt.epochs, base_lr, 0.9)

            elif phase == 'test':
                df_test = pd.concat(
                    [
                        df_test,
                        pd.DataFrame(
                            [
                                {
                                    "case": opt.case,
                                    "epoch": epoch,
                                    "wall_time": time.time() - since,
                                    "test_dice": -epoch_loss,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                df_test.to_csv(df_path_test, index=False)

            else:
                df_train = pd.concat(
                    [
                        df_train,
                        pd.DataFrame(
                            [
                                {
                                    "case": opt.case,
                                    "epoch": epoch,
                                    "wall_time": time.time() - since,
                                    "train_dice": -epoch_loss,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                df_train.to_csv(df_path_train, index=False)

        # Checkpoints / stopping
        if epoch == opt.epochs:
            torch.save(model.state_dict(), save_path.format("final"))
            continue_training = False

        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path.format(str(epoch)))

    with open(os.path.join(folder_case, "info.pickle"), "wb") as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    torch.cuda.synchronize()
    time_elapsed = time.perf_counter() - since
    logger.info(f"[INFO] Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logger.info(f"[INFO] Best validation epoch is {best_epoch}")


def main():
    opt = parsing_data()
    set_determinism(seed=opt.seed)

    # Folders
    fold_dir = os.path.join(opt.model_dir, opt.case, str(opt.learning_rate), str(opt.spacial), opt.comment)
    fold_dir_model = os.path.join(fold_dir, "models")
    os.makedirs(fold_dir_model, exist_ok=True)

    save_path = os.path.join(fold_dir_model, "./CP_{}.pth")

    if opt.path_labels is None:
        opt.path_labels = opt.path_data

    logger = create_logger(os.path.join(fold_dir))
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Case: {opt.case}")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Initial lr: {opt.learning_rate}")
    logger.info(f"Total number of epochs: {opt.epochs}")

    # GPU check
    if not torch.cuda.is_available():
        raise logger.error("[INFO] No GPU found")

    logger.info("[INFO] GPU available.")
    device = torch.device("cuda:0")

    split_path = os.path.join(opt.synthesizer_dir, 'split.csv')
    df_split = pd.read_csv(split_path,header =None)
    list_file_inference = df_split[df_split[1].isin(['inference'])][0].tolist()
    assert opt.case+'-' in list_file_inference, f'Synthesizor was trained with {opt.case} - Forbidden'
    print(f'[INFO] Check passed: synthesizor was not trained with {opt.case}')

    # load case mri
    images_modalities = get_coregistered_mr_images(opt.path_data, opt.case)
    print(opt.drop_modalities)
    print(images_modalities)
    print([k[0] for k in images_modalities])
    modalities = [k[0] for k in images_modalities if k[0] != opt.drop_modalities[0]]
    print(modalities)
    total_subsets_mr = list(
        chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1))
    )
    print(total_subsets_mr)

    mri_vols = {}
    for mod in modalities+['target']:
        path = glob.glob(os.path.join(opt.path_data, opt.case, f'{opt.case}-{mod}**.nii.gz'))[0]
        # if mod == 'target':
        #     path = glob.glob(os.path.join(opt.path_data, opt.case, f'{opt.case}-{mod}_n2.nii.gz'))[0]
        img = nib.load(path)
        img_affine = torch.from_numpy(img.affine).to(torch.float32)
        data = img.get_fdata() 
        inv_affine = torch.linalg.inv(img_affine)
        vol_np = np.transpose(data, (2, 1, 0))
        vol_t = torch.from_numpy(vol_np).to(torch.float32).unsqueeze(0).unsqueeze(0)
        mri_vols[mod] = {
            'volume': vol_t,
            'affine': img.affine,
            'inv_affine': inv_affine
        }
    
    # load case target
    # path_target = os.path.join(opt.path_data, opt.case, f'{opt.case}-target_n2.nii.gz')
    path_target = os.path.join(opt.path_data, opt.case, f'{opt.case}-target.nii.gz')
    img_target = nib.load(path_target)
    data_target = img_target.get_fdata()

    ultrasounds = [k for k in os.listdir(opt.path_data) if "Case" in k and opt.case not in k] # len = 102

    # extract surface volume
    path_img_strip = glob.glob(os.path.join(opt.path_strip, opt.case, f"{opt.case}-**_mask.nii.gz"))[0]
    data_noncerebrum = (nib.load(path_img_strip).get_fdata() == 0)
    border = find_boundaries(data_noncerebrum, mode="inner")

    border_pixels = np.stack(np.where(border),-1)
    new_arr = vox_to_world_many(img_target.affine, border_pixels)

    affine_path = os.path.join('../data/registration/mni', opt.case, f"{opt.case}-mri-to-mni-Syn0GenericAffine.mat")
    sitk_affine = sitk.ReadTransform(affine_path)

    mni_seg_img = sitk.ReadImage('../data/mni/mni_brain_surface.nii.gz.seg.nrrd')
    mni_surface_mask = sitk.GetArrayFromImage(mni_seg_img)[..., 0] > 0 # (z,y,x)

    non_viable_surface_mask = sitk.GetArrayFromImage(sitk.ReadImage('../data/mni/refined_non_viable_surface_mask2.nii.gz')) > 0 # (z,y,x)
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

    img_mri = sitk.ReadImage(glob.glob(os.path.join(opt.path_data, opt.case, f'{opt.case}-**_ref.nii.gz'))[0])
    radius_mm = np.sqrt(img_mri.GetSpacing()[0]**2 + img_mri.GetSpacing()[1]**2 + img_mri.GetSpacing()[2]**2)

    idx = tree.query_ball_point(new_arr, r=radius_mm) # for every point in new_arr, idx returns the points in the tree which are within distance r
    mask = np.array([len(n) == 0 for n in idx])
    filtered_arr = new_arr[mask]

    # points on brain surface we can query
    surface_tree = cKDTree(filtered_arr) if len(filtered_arr) > 0 else None

    # extract target information
    target_mask = np.argwhere(data_target > 0)
    target_world = vox_to_world_many(img_target.affine, target_mask)

    # points belonging to target in world coordinates
    TUMOR_POINTS_3D = np.array(target_world)

    m_2 = [k.mean() for k in np.where(data_target > 0)]
    m_2_noisy = m_2 + np.random.normal(loc=0.0, scale=3.0, size=(3))

    # target CoM in world coordinates
    m_2_noisy = vox_to_world(img_target.affine, m_2_noisy)
    m_2 = vox_to_world(img_target.affine, m_2)

    d_2 = np.linalg.norm(filtered_arr - m_2_noisy, axis=1)
    weights = np.exp(-d_2)
    weights[weights<0] = 0
    weights /= weights.sum()

    # Synthesizer
    print("[INFO] Building hierarchical multi-modal synthesizer")  
    synthesizer = MHVAE2D(
        modalities=opt.modalities,   
        base_num_features=opt.base_features,   
        num_pool=opt.pools,   
        original_shape=opt.spatial_shape[:2],
        max_features=opt.max_features,
        with_residual=opt.no_res,
        with_se=opt.no_se,
        nb_finalblocks=opt.nb_finalblocks,
        nfeat_finalblock=opt.nfeat_finalblock,
        ).eval().to(device)
    
    save_path_syn = os.path.join(opt.synthesizer_dir, 'models', './CP_main_{}.pth')
    assert os.path.exists(save_path_syn.format(opt.synthesizer_epoch)), f"Model weights not found: {save_path_syn.format(opt.synthesizer_epoch)}"

    synthesizer.load_state_dict(torch.load(save_path_syn.format(opt.synthesizer_epoch)))
    print(f"Loading synthesizer from {save_path_syn.format(opt.synthesizer_epoch)}")

    # UNet Model
    logger.info("[INFO] Building model")

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

    # checkpoint_path = './models/bratious/0.01/False/remind-100-0.01/models/CP_final.pth'
    # assert os.path.isfile(checkpoint_path), f"no checkpoint found {checkpoint_path}"
    # model.load_state_dict(torch.load(checkpoint_path))
    # logger.info("[INFO] Pre-trained model has been loaded")

    # Losses
    criterion = DC(NB_CLASSES)
    soft_criterion = DC_SOFT(NB_CLASSES)

    paths_dict = {split: [] for split in PHASES[1:]}
    nb_valimages = opt.nb_val

    def add_subject(split, subject, path_data):
        img_path = os.path.join(path_data, subject)
        lab_flnm = subject.split("_")[0] + '_target.nii.gz'
        lab_path = os.path.join(path_data, lab_flnm)
        if split == 'test':
            img_path = f'{path_data}/imgs/reslice{subject}_crop.nii.gz'
            lab_path = f'{path_data}/gt/reslice{subject}_crop.nii.gz'
        if os.path.exists(img_path) and os.path.exists(lab_path):
            paths_dict[split].append({"img": img_path, "seg": lab_path})


    output_path = os.path.join(opt.output_val, opt.case)
    os.makedirs(output_path, exist_ok=True)

    ultrasounds_subset = iter(np.random.choice(ultrasounds, nb_valimages + 10, replace=False).tolist())

    count = 0
    while count < nb_valimages:
        output, modalities_set, subject = generate_us_sweep(total_subsets_mr, ultrasounds_subset, filtered_arr, weights, TUMOR_POINTS_3D, m_2, surface_tree, mri_vols, device, output_path, radius_mm, mode='validation', case=opt.case)
        ultrasounds.remove(subject)
        listToStr = "-".join([str(elem) for elem in modalities_set])
        flnm = f"{opt.case}-{subject}-{listToStr}"
        _ = synthesize_us_sweep(synthesizer, modalities_set, output, device, opt.type_normalization, os.path.join(output_path, f'{flnm}'+'_{}.nii.gz'), mode='validation')
        count += 1

    ultrasounds = itertools.cycle(ultrasounds)

    for subject in os.listdir(output_path):
        if not subject.replace('.nii.gz', '').split('_')[-1] in ['0.3','0.5','0.7','1.0']:
            continue
        add_subject("validation", subject, output_path)

    add_subject("test", opt.case, opt.path_test)

    for split in PHASES[1:]:
        logger.info(f"Nb patients in {split} data: {len(paths_dict[split])}")

    train(paths_dict, synthesizer, model, criterion, soft_criterion, criterion_ce, soft_ce, device, save_path, logger, opt, mod_vols=mri_vols, total_subsets_mr=total_subsets_mr, ultrasounds=ultrasounds, filtered_arr=filtered_arr, weights=weights, m_2=m_2, tumor_points_3d=TUMOR_POINTS_3D, surface_tree=surface_tree, radius_mm=radius_mm)


def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using extreme points as supervision"
    )

    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--synthesizer_dir", type=str, help="Path to the synthesizer directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of the batch size (default: 64)")
    parser.add_argument("--case", type=str, default="Case112")
    parser.add_argument("--path_data", type=str, default="../data/robustmislite/Training_R_low/")
    parser.add_argument("--path_labels", type=str, default=None, help="Path to the labels")
    parser.add_argument("--path_test", type=str, default='./miccai2024_data/test_set/remind', help="Path to the test data")
    parser.add_argument("--path_strip", type=str, default=None, help="Path to the skull stripped mri volumes")
    parser.add_argument("--comment", type=str, default="", help="Experiment comment/tag")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Total number of epochs")
    parser.add_argument("--nb_val", type=int, default=20, help="Total validation images")
    parser.add_argument("--output_val", type=str, default='./miccai2024_data/synthetic/validation_otf', help="Path to the validation data")
    parser.add_argument("--pretrain_epoch", type=str, default="final")
    parser.add_argument("--synthesizer_epoch", type=str, default=1000)
    parser.add_argument("--spacial", action="store_true", help="Aug spacial enabled")
    parser.add_argument('--modalities',
                    type=str,
                    nargs="+",
                    default=['us', 't2', 'cet1', 'flair'])
    parser.add_argument('--drop_modalities',
                    type=str,
                    nargs="+",
                    default=None)
    parser.add_argument('--type_normalization',
                        type=str,
                        default='min-max',
                        help='Type of normalization')
    parser.add_argument('--base_features',
                    type=int,
                    default=16,
                    help='Latent Dimensionality highest level (divided by 2)')
    parser.add_argument('--pools',
                    type=int,
                    default=6,
                    help='Number of latent representation below z_1')
    parser.add_argument('--spatial_shape',
                    type=int,
                    nargs="+",
                    default=(192,192))
    parser.add_argument('--max_features',
                    type=int,
                    default=128,
                    help='Max Latent Dimensionality')
    parser.add_argument('--no_res', 
                        action='store_false', 
                        help='Residual connection disabled')
    parser.add_argument('--no_se',
                        action='store_false', 
                        help='Squeeze and Excitation disabled')
    parser.add_argument('--nb_finalblocks',
                    type=int,
                    default=6,
                    help='Number of blocks from z_1 to x_i')
    parser.add_argument('--nfeat_finalblock',
                    type=int,
                    default=8,
                    help='Number of channels in blocks from z_1 to x_i')
    parser.add_argument('--seed', type=int, default=2)
    

    return parser.parse_args()


if __name__ == "__main__":
    main()
