import argparse
import os
import pickle
import time
import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm
from monai.utils import set_determinism

from networks.unet import UNet2D
from dataloaders.dataloader_miccai24 import get_training_loader, get_validation_loader
from utilities.losses import DC, DC_SOFT
from utilities.utils import create_logger, infinite_iterable, poly_lr, draw_curve

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


def train(paths_dict, model, criterion, soft_criterion, device, save_path, logger, opt):
    logger.info("[INFO] Starting training")
    logger.info(f"Case {opt.case}")
    since = time.time()

    # const_iter = len([k for k in os.listdir(f"./miccai2024_data/synthetic/remind-10/{opt.case}/data_us")]) - 20
    # print('const_iter ', const_iter)

    # Dataloaders
    ind_batch = {
        "training": np.arange(0, min(1000, len(paths_dict["training"])), opt.batch_size),
        "validation": np.arange(0, min(1000, len(paths_dict["validation"])), 1),
        "test": np.arange(0, len(paths_dict["test"]), 1)
    }

    dataloaders = {
        "training": infinite_iterable(
            get_training_loader(
                paths_dict["training"],
                batch_size=opt.batch_size,
                num_workers=WORKERS,
                use_spacial_augmentation=opt.spacial,
                cache=CACHE,
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
            get_validation_loader(
                paths_dict["test"],
                batch_size=1,
                num_workers=0,
                cache=CACHE,
            )
        )
    }

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
                data_time = time.time()
                batch = next(dataloaders[phase])

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

                data_time = time.time() - data_time
                data_time_avg.append(data_time)

                if epoch == 1:
                    visualize_batch(batch, os.path.join(folder_case, f"debug_batch_{phase}.png"))

                train_time = time.time()
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

                train_time = time.time() - train_time
                train_time_avg.append(train_time)

                if epoch % 1 == 0:
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

    time_elapsed = time.time() - since
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

    # Split data
    list_file = [k.split(".nii")[0] for k in os.listdir(opt.path_data) if ".nii" in k]
    list_file = sorted(set(list_file))
    np.random.shuffle(list_file)

    paths_dict = {split: [] for split in PHASES}
    nb_valimages = opt.nb_val

    def add_subject(split, subject):
        img_path = os.path.join(opt.path_data, subject + ".nii.gz")
        lab_flnm = subject.split("_")[0] + "_seg.nii.gz"
        lab_path = os.path.join(opt.path_labels, lab_flnm)
        if split == 'test':
            img_path = f'./miccai2024_data/test_set/remind/imgs/reslice{subject}_crop.nii.gz'
            lab_path = f'./miccai2024_data/test_set/remind/gt/reslice{subject}_crop.nii.gz'
        if os.path.exists(img_path) and os.path.exists(lab_path):
            paths_dict[split].append({"img": img_path, "seg": lab_path})

    for subject in list_file[:-nb_valimages]:
        add_subject("training", subject)

    for subject in list_file[-nb_valimages:]:
        add_subject("validation", subject)

    for split in PHASES:
        logger.info(f"Nb patients in {split} data: {len(paths_dict[split])}")

    add_subject("test", opt.case)
    logger.info(f"Nb patients in test data: {len(paths_dict['test'])}")

    # Model
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

    # checkpoint_path = os.path.join(opt.model_dir, "brats", "models", "./CP_{}.pth").format(opt.pretrain_epoch)
    # assert os.path.isfile(checkpoint_path), f"no checkpoint found {checkpoint_path}"
    # model.load_state_dict(torch.load(checkpoint_path))
    # logger.info("[INFO] Pre-trained model has been loaded")

    # Losses
    criterion = DC(NB_CLASSES)
    soft_criterion = DC_SOFT(NB_CLASSES)

    train(paths_dict, model, criterion, soft_criterion, device, save_path, logger, opt)


def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using extreme points as supervision"
    )

    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of the batch size (default: 64)")
    parser.add_argument("--case", type=str, default="Case112")
    parser.add_argument("--path_data", type=str, default="../data/robustmislite/Training_R_low/")
    parser.add_argument("--path_labels", type=str, default=None, help="Path to the labels")
    parser.add_argument("--comment", type=str, default="", help="Experiment comment/tag")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Total number of epochs")
    parser.add_argument("--nb_val", type=int, default=20, help="Total validation images")
    parser.add_argument("--pretrain_epoch", type=str, default="final")
    parser.add_argument("--spacial", action="store_true", help="Aug spacial enabled")
    parser.add_argument('--seed', type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    main()
