from PIL import Image
import nibabel as nib
import numpy as np
import os


def flip(array: np.ndarray) -> np.ndarray:
    """Flip along axis 0."""
    return np.flip(array, 0)


def add_contour(In: Image.Image, Seg: Image.Image, Color=(255, 255, 0)) -> Image.Image:
    """
    Add a segmentation contour to a 2D image.

    Args:
        In: PIL Image (RGB recommended)
        Seg: binary segmentation (PIL Image; non-zero pixels are foreground)
        Color: RGB value for contour color

    Returns:
        A copy of In with the contour drawn.
    """
    Out = In.copy()

    # PIL Image.size is (width, height). Keep original variable names to minimize change.
    H, W = In.size

    for i in range(H):
        for j in range(W):
            seg_ij = Seg.getpixel((i, j)) != 0
            if not seg_ij:
                continue

            on_border = (i == 0 or i == H - 1 or j == 0 or j == W - 1)
            if on_border:
                Out.putpixel((i, j), Color)
                continue

            # Contour pixel if any 4-neighbor is background
            seg_up = Seg.getpixel((i - 1, j)) != 0
            seg_down = Seg.getpixel((i + 1, j)) != 0
            seg_left = Seg.getpixel((i, j - 1)) != 0
            seg_right = Seg.getpixel((i, j + 1)) != 0

            if not (seg_up and seg_down and seg_left and seg_right):
                Out.putpixel((i, j), Color)

    return Out


def add_filling(In: Image.Image, Seg: Image.Image, Color=(255, 255, 0)) -> Image.Image:
    """
    Fill the segmentation region (and also draw its contour, as in the original code).

    Args:
        In: PIL Image
        Seg: binary segmentation (PIL Image)
        Color: RGB value for fill/contour color

    Returns:
        A copy of In with the segmentation region colored.
    """
    Out = In.copy()
    H, W = In.size

    for i in range(H):
        for j in range(W):
            seg_ij = Seg.getpixel((i, j)) != 0
            if seg_ij:
                # Fill
                Out.putpixel((i, j), Color)

            on_border = (i == 0 or i == H - 1 or j == 0 or j == W - 1)
            if on_border:
                if seg_ij:
                    Out.putpixel((i, j), Color)
                continue

            if seg_ij:
                seg_up = Seg.getpixel((i - 1, j)) != 0
                seg_down = Seg.getpixel((i + 1, j)) != 0
                seg_left = Seg.getpixel((i, j - 1)) != 0
                seg_right = Seg.getpixel((i, j + 1)) != 0

                if not (seg_up and seg_down and seg_left and seg_right):
                    Out.putpixel((i, j), Color)

    return Out


# ------------------------
# Configuration
# ------------------------
alpha = 255
colors_int = [
    (0, 255, 0, alpha),
    (0, 0, 255, alpha),
    (0, 255, 255, alpha),
    (255, 0, 255, alpha),
    (255, 0, 0, alpha),
    (85, 170, 127, alpha),
    (170, 0, 0, alpha),
    (255, 170, 127, alpha),
]
colors_int = colors_int[::-1]  # better for visualization

save_gif = True  # if you want to generate a GIF afterwards
output_folder = './experiments/test_set/visualizations'

path_predictions = './experiments/test_set'
modality = {"Case027": "t2"}
methods_eval = ["brats_unet"]
protocols_eval = ["remind"]
path_us = './experiments/test_set/remind/imgs/reslice{}_crop.nii.gz'
path_gt = './experiments/test_set/{}/gt/reslice{}_crop.nii.gz'

if save_gif:
    import glob
    import imageio.v2 as imageio


# ------------------------
# Main loop
# ------------------------
cases = list(modality.keys())
cases = ['Case027','Case045', 'Case074', 'Case085', 'Case099', 'Case103','Case112']

for case in cases:
    data_us = nib.load(path_us.format(case)).get_fdata()
    total_slice = data_us.shape[-1]  # (unused, kept)

    for z_slice in np.arange(0, data_us.shape[-1] - 1, 1):
        us_slice = flip(data_us[..., z_slice])

        img_us = Image.fromarray(us_slice / np.max(us_slice) * 255)
        img_us = img_us.convert("RGB")

        for i, protocol in enumerate(protocols_eval):
            for method in methods_eval:
                img_output = img_us

                # Ground truth
                path_file_gt = path_gt.format(protocol, case)
                gt_slice = flip(nib.load(path_file_gt).get_fdata()[..., z_slice])
                gt_slice = Image.fromarray(gt_slice)

                # Prediction
                path_folder = os.path.join(path_predictions, protocol, method)
                path_file_pred = os.path.join(path_folder, f"reslice{case}_crop.nii.gz")
                pred_slice = flip(nib.load(path_file_pred).get_fdata()[..., z_slice])
                pred_slice = Image.fromarray(pred_slice)

                # Overlay: fill -> blend -> contours
                img_output = add_filling(img_output, pred_slice, Color=colors_int[0])  # fill
                img_output = Image.blend(img_us, img_output, 0.15)
                img_output = add_contour(img_output, pred_slice, Color=colors_int[0])     # pred contour
                img_output = add_contour(img_output, gt_slice, Color=colors_int[i + 1])  # gt contour

                os.makedirs(f"{output_folder}/{case}/{protocol}/{method}", exist_ok=True)
                img_output.save(f"{output_folder}/{case}/{protocol}/{method}/{z_slice:03d}.png")

    if save_gif:
        for i, protocol in enumerate(protocols_eval):
            for method in methods_eval:
                png_dir = f"{output_folder}/{case}/{protocol}/{method}"
                out_gif = f"{output_folder}/{case}/{protocol}/{method}/pred_miccai.gif"

                pngs = glob.glob(os.path.join(png_dir, f"*.png"))
                pngs = sorted(pngs)

                frames = [imageio.imread(p) for p in pngs]
                frames += frames[-2::-1]  # reverse (avoid duplicate midpoint)

                fps = 10
                duration = 1.0 / fps
                imageio.mimsave(out_gif, frames, fps=fps, loop=0)

                print("Saved GIF:", out_gif)
