import logging
import os
import matplotlib.pyplot as plt
import nibabel as nib
import torch

import logging

import random
import numpy as np

import warnings

import torch

_seed = None
_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
NP_MAX = np.iinfo(np.uint32).max
MAX_SEED = NP_MAX + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32


def set_determinism(
    seed: int | None = NP_MAX,
    use_deterministic_algorithms: bool | None = None,
   
) -> None:
    """
    Set random seed for modules to enable or disable deterministic training.

    Args:
        seed: the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        use_deterministic_algorithms: Set whether PyTorch operations must use "deterministic" algorithms.
        additional_settings: additional settings that need to set random seed.

    Note:

        This function will not affect the randomizable objects in :py:class:`monai.transforms.Randomizable`, which
        have independent random states. For those objects, the ``set_random_state()`` method should be used to
        ensure the deterministic behavior (alternatively, :py:class:`monai.data.DataLoader` by default sets the seeds
        according to the global random state, please see also: :py:class:`monai.data.utils.worker_init_fn` and
        :py:class:`monai.data.utils.set_rnd`).
    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)


    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(use_deterministic_algorithms)
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")


def create_logger(folder):
    """Create a logger to save logs."""
    compt = 0
    while os.path.exists(os.path.join(folder,f"logs_{compt}.txt")):
        compt+=1
    logname = os.path.join(folder,f"logs_{compt}.txt")
    
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(logname, mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    return logger 
    

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """Learning rate policy used in nnUNet."""
    return initial_lr * (1 - epoch / max_epochs)**exponent


def infinite_iterable(i):
    while True:
        yield from i

    
def draw_curve(infos):
    if len(infos['epochs'])>0:
        fig = plt.figure(figsize=(12, 10))
        ax0 = fig.add_subplot(311, title="Training")
        ax1 = fig.add_subplot(312, title="Validation")
        ax2 = fig.add_subplot(313, title="Test")

        ax0.plot(infos['epochs'], infos['training'], 'bo-', label='Multi-Res Dice Loss')
        ax1.plot(infos['epochs'], infos['validation'], 'ro-', label='Dice Score')
        ax2.plot(infos['epochs'], infos['test'], 'ro-', label='Dice Score')
        
        ax1.legend()
        ax0.legend()
        ax2.legend()
        fig.savefig(infos['path'])

        
def save(pred, affine, path):
    pred = (255*(pred+1)/2).int()
    pred = pred.permute(2,3,0,1).squeeze().cpu().numpy()
    nib.Nifti1Image(pred, affine).to_filename(path)