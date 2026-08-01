import argparse
import os
import pickle
import time
import itertools
import monai
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm
from monai.utils import set_determinism

from networks.unet import UNet2D
from dataloaders.dataloader_otf import *
from utilities.losses import *
from utilities.utils import create_logger, infinite_iterable, poly_lr, draw_curve
from utilities.sweep_generator import SyntheticSweepGenerator, SyntheticSweepConfig, SynthesisConfig

import monai.transforms.compose
import monai.transforms.transform

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

def build_validation_and_test_paths(output_path: str, case: str, path_test: str):
    paths_dict = {"validation": [], "test": []}

    def add_subject(split, subject, path_data):
        img_path = os.path.join(path_data, subject)
        lab_flnm = subject.split("_")[0] + f"_target.nii.gz"
        lab_path = os.path.join(path_data, lab_flnm)

        if split == "test":
            img_path = f"{path_data}/imgs/reslice{subject}_crop.nii.gz"
            lab_path = f"{path_data}/gt/reslice{subject}_crop.nii.gz"

        if os.path.exists(img_path) and os.path.exists(lab_path):
            paths_dict[split].append({"img": img_path, "seg": lab_path})

    for subject in os.listdir(output_path):
        suffix = subject.replace(".nii.gz", "").split("_")[-1]
        if suffix not in ["0.3", "0.5", "0.7", "1.0"]:
            continue
        add_subject("validation", subject, output_path)

    add_subject("test", case, path_test)
    return paths_dict

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


def train(
        paths_dict,
        sweep_generator,
        context,
        ultrasounds,
        synthesizer,
        model,
        criterion,
        soft_criterion,
        device,
        save_path,
        logger,
        opt
    ):    
    logger.info("[INFO] Starting training")
    logger.info(f"Case {opt.case}")
    torch.cuda.synchronize()
    since = time.perf_counter()

    const_iter = len([k for k in os.listdir(f"./experiments/synthetic/n1-10-2/{opt.case}/data_us")]) - 20
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
                batch_size=opt.batch_size,
                num_workers=WORKERS,
                sweep_generator=sweep_generator,
                context=context,
                ultrasounds=ultrasounds,
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

                    modalities_set = random.choice(context.total_subsets_mr)

                    output = sweep_generator.render_sweep_volumes(
                        context=context,
                        modalities_set=modalities_set,
                        ps=Ps,
                        d_vecs=d_vecs,
                        i_vecs=i_vecs,
                        p_us=P_origin,
                        fov_slice=fov_slice,
                        device=device,
                        include_target=True,
                        y_grid=Y_grid,
                        x_grid=X_grid,
                    )

                    torch.cuda.synchronize()
                    data_train_generation = time.perf_counter() - data_train_generation
                    data_time_gen_avg.append(data_train_generation)
                    torch.cuda.synchronize()
                    data_train_synthesis = time.perf_counter()
                    pred = sweep_generator.synthesize_us_sweep_in_memory(
                        synthesizer=synthesizer,
                        modalities=modalities_set,
                        output=output,
                        device=device,
                        type_normalization=opt.type_normalization,
                        mode="training",
                    )
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

    fold_dir = os.path.join(opt.model_dir, opt.case, str(opt.learning_rate), str(opt.spacial), opt.comment)
    fold_dir_model = os.path.join(fold_dir, "models")
    os.makedirs(fold_dir_model, exist_ok=True)

    save_path = os.path.join(fold_dir_model, "CP_{}.pth")

    logger = create_logger(os.path.join(fold_dir))
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Case: {opt.case}")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Initial lr: {opt.learning_rate}")
    logger.info(f"Total number of epochs: {opt.epochs}")

    if not torch.cuda.is_available():
        raise RuntimeError("[INFO] No GPU found")

    device = torch.device("cuda:0")
    logger.info("[INFO] GPU available.")

    # ------------------------------------------------------------------
    # Sweep generator + case context
    # ------------------------------------------------------------------
    sweep_cfg = SyntheticSweepConfig(
        path_data_mri=opt.path_data,
        path_skull_strip=opt.path_strip,
        out_h=192,
        out_w=192,
        dx_mm=0.5,
        dy_mm=0.5,
    )
    sweep_generator = SyntheticSweepGenerator(sweep_cfg)

    context = sweep_generator.prepare_case_context(
        case=opt.case,
        annotator=opt.annotator,
        drop_modalities=opt.drop_modalities,
        target_name=f"target",
    )

    # ------------------------------------------------------------------
    # Synthesis leakage check + synthesizer model
    # ------------------------------------------------------------------
    sweep_generator._check_case_not_seen_during_training(
        model_dir=opt.synthesizer_dir,
        case=opt.case
    )

    syn_cfg = SynthesisConfig(
        model_dir=opt.synthesizer_dir,
        case=opt.case,
        type_normalization=opt.type_normalization,
        base_features=opt.base_features,
        pools=opt.pools,
        spatial_shape=tuple(opt.spatial_shape),
        max_features=opt.max_features,
        no_res=opt.no_res,
        no_se=opt.no_se,
        nb_finalblocks=opt.nb_finalblocks,
        nfeat_finalblock=opt.nfeat_finalblock,
        epoch_inf=opt.synthesizer_epoch,
        modalities=opt.modalities,
    )

    synthesizer = sweep_generator._build_synthesis_model(device, syn_cfg).eval()
    sweep_generator._load_synthesis_weights(synthesizer, syn_cfg)

    # ------------------------------------------------------------------
    # Segmentation model
    # ------------------------------------------------------------------
    logger.info("[INFO] Building segmentation model")

    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    model = UNet2D(
        input_channels=1,
        base_num_features=32,
        num_classes=NB_CLASSES,
        num_pool=5,
        conv_op=nn.Conv2d,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs=norm_op_kwargs,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ).to(device)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    criterion = DC(NB_CLASSES)
    soft_criterion = DC_SOFT(NB_CLASSES)

    # ------------------------------------------------------------------
    # Build fixed validation set
    # ------------------------------------------------------------------
    output_path = os.path.join(opt.output_val, opt.case)
    os.makedirs(output_path, exist_ok=True)

    available_us = sweep_generator.generate_fixed_validation_dataset(
        context=context,
        output_path=output_path,
        synthesizer=synthesizer,
        num_samples=opt.nb_val,
        device=device,
        type_normalization=opt.type_normalization,
    )

    # ------------------------------------------------------------------
    # Build validation/test path dictionaries
    # ------------------------------------------------------------------
    paths_dict = build_validation_and_test_paths(
        output_path=output_path,
        case=opt.case,
        path_test=opt.path_test,
    )

    for split in ["validation", "test"]:
        logger.info(f"Nb patients in {split} data: {len(paths_dict[split])}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    ultrasounds_iter = itertools.cycle(available_us)

    train(
        paths_dict=paths_dict,
        sweep_generator=sweep_generator,
        context=context,
        ultrasounds=ultrasounds_iter,
        synthesizer=synthesizer,
        model=model,
        criterion=criterion,
        soft_criterion=soft_criterion,
        device=device,
        save_path=save_path,
        logger=logger,
        opt=opt,
    )


def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using extreme points as supervision"
    )

    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--synthesizer_dir", type=str, help="Path to the synthesizer directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of the batch size (default: 64)")
    parser.add_argument("--case", type=str, default="Case112")
    parser.add_argument("--annotator", type=str, default="n1")
    parser.add_argument("--path_data", type=str, default="../data/robustmislite/Training_R_low/")
    parser.add_argument("--path_labels", type=str, default=None, help="Path to the labels")
    parser.add_argument("--path_test", type=str, default='./experiments/test_set/n1', help="Path to the test data")
    parser.add_argument("--path_strip", type=str, default=None, help="Path to the skull stripped mri volumes")
    parser.add_argument("--comment", type=str, default="", help="Experiment comment/tag")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Total number of epochs")
    parser.add_argument("--nb_val", type=int, default=20, help="Total validation images")
    parser.add_argument("--output_val", type=str, default='./experiments/synthetic/validation_otf', help="Path to the validation data")
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
