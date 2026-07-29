import os

from itertools import chain, combinations

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from utilities.generation import (
    add_center_norm_us,
    add_corners_mr,
    find_boundaries,
    get_corners_us,
    get_mr_images,
    get_tumor_landmarks,
    mask,
    quantisize,
    resample_seg,
    rigid_sitk,
    zero_mean,
    zeros_like,
    save_all,
    create_reslicing
)
import argparse
from utilities.utils import set_determinism

ALL_CASES = ["Case011", "Case025", "Case027", "Case045", "Case052", "Case056", "Case070", "Case074", "Case085", "Case099", "Case103", "Case112", "Case114"]

parser = argparse.ArgumentParser(
                    prog='Generatedata',
                    description='This script creates ultrasound sweeps on MRI data.')

parser.add_argument('--K', type=int, default=1, help='Number of K sweeps')
parser.add_argument('--annotator', type=str, default="remind", help='Annotator')
parser.add_argument('--case', type=str, default="Case112", help='Case')
parser.add_argument('--seed', type=int, default=0, help='Seed')


args = parser.parse_args()

# --------------------
# Hyperparameters
# --------------------
NB_cases = args.K
debug = False
save_landmark = True
annotator = args.annotator
full_modalities = ["t2", "cet1", "flair"]
case = args.case
assert case in ALL_CASES, f"Error {case} in not in {ALL_CASES}"
seed = args.seed

path_output = "./experiments/synthetic/{}-{}-{}/{}/{}/{}"

# Folders from data processing
output_path_reg_mr = "./data/registration/rigid"
output_path_reg_us = "./data/registration/affine/"
path_imgs = pd.read_csv("./data/refs_files.csv", index_col=0).T

path_us_space = "./data/coregistered/us-space/"
path_us_all = os.path.join(path_us_space, "{0}/{0}-us.nii.gz") 
path_folder = "./data/nrrd"

# Folders from MICCAI
path_strip = "./experiments/skullstripping_hdbet/"
path_seg = f"./experiments/training_mrlabels/{annotator}"
us_mask_probe = "./experiments/crop_mask/{}-mask.nii.gz"


set_determinism(seed=seed)


# --------------------
# 1) Reference MR + tumor label in MR-ref space
# --------------------
path_premr = os.path.join(path_folder, case, "Preop-MR")
img_path_mr = os.path.join(path_premr, path_imgs[case]["mr"])
mr_flnm_or = path_imgs[case]["mr"].replace(".nrrd", "")

img_mr_ref = sitk.ReadImage(img_path_mr)

# Load tumor segmentation (nrrd) for this case
segs = [k for k in os.listdir(path_seg) if case in k]
assert len(segs) == 1, "error more than 1 seg available"
seg_path = os.path.join(path_seg, segs[0])
seg = sitk.ReadImage(seg_path)

# Figure out transform that maps label space -> MR-ref space
mr_ref_label = segs[0].replace(".nrrd", "").replace("SEG-tumor-", "")
if mr_flnm_or in mr_ref_label:
    transform_label = sitk.Transform()
else:
    transform_label_filnm = os.path.join(
        output_path_reg_mr, case, f"{case}-{mr_flnm_or}-to-{mr_ref_label}.tfm"
    )
    transform_label = sitk.ReadTransform(transform_label_filnm).GetInverse()

path_label = f"./experiments/synthetic/label-{annotator}/label-mr-space/{case}/seg.nii.gz"
if not os.path.exists(path_label):
    new_seg = resample_seg(seg, img_mr_ref, transform_label)
    os.makedirs(os.path.dirname(path_label), exist_ok=True)
    sitk.WriteImage(new_seg, path_label)

# --------------------
# 2) Prepare output folders + candidate US list
# --------------------
for folder in ["data", "data_reslice", "landmarks", "reg", "label", "tracking"]:
    os.makedirs(path_output.format(annotator, NB_cases, seed, case, folder, ""), exist_ok=True)

ultrasounds = [k for k in os.listdir(path_us_space) if "Case" in k and case not in k]

# --------------------
# 3) Extract tumor-related landmarks from MR
# --------------------
path_img_strip = os.path.join(path_strip, f"{case}/{case}-{mr_flnm_or}_mask.nii.gz")
data_noncerebrum = (nib.load(path_img_strip).get_fdata() == 0)
border = find_boundaries(data_noncerebrum, mode="inner")

data_label = nib.load(path_label).get_fdata()
data_coretumor = data_label > 0
data_whole_tumor = data_label > 0
affine_img = nib.load(path_label).affine

points_mr, points_mr_ind = get_tumor_landmarks(
    border, data_whole_tumor, data_coretumor, affine_img
)

# MR modalities + all non-empty subsets
images_modalities = get_mr_images(path_folder, case, threed_only=False)
modalities = [k[0] for k in images_modalities]
total_subsets_mr = list(
    chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1))
)

print(total_subsets_mr)

path_output_data = path_output.format(annotator, NB_cases, seed, case, "data", "")

# --------------------
# 4) For each modality subset: generate NB_cases synthetic pairs (MR->US)
# --------------------
for modalities_set in total_subsets_mr:
    ultrasounds_subset = iter(np.random.choice(ultrasounds, NB_cases + 10, replace=False).tolist())
    missing_modalities = [m for m in full_modalities if m not in modalities_set]

    count = 0
    while count < NB_cases:
        us = next(ultrasounds_subset)
        try:
            # Load US + probe mask
            img_us = nib.load(path_us_all.format(us))
            data_us = img_us.get_fdata()
            affine_us = img_us.affine
            data_maskus = nib.load(us_mask_probe.format(us)).get_fdata()

            # US landmarks (corners)
            points_us, points_us_ind = get_corners_us(data_us, data_maskus, affine_us)

            # Add MR corners based on US + MR border
            points_mr, points_mr_ind = add_corners_mr(
                points_mr, points_mr_ind, points_us, border, affine_img
            )

            if debug:
                print(us)
                save_all([points_mr, points_mr_ind, points_us, points_us_ind])

            # Add normalized center landmark in US
            points_us, points_us_ind = add_center_norm_us(
                points_us,
                points_us_ind,
                points_mr,
                affine_us,
                ratio_pixel=0.5,
                largest_dim=191,
            )

            # Rigid alignment from US landmarks -> MR landmarks
            ar_points_us = np.stack([points_us[k] for k in ["corner_1", "corner_2", "center", "component"]], 1)
            ar_points_mr = np.stack([points_mr[k] for k in ["corner_1", "corner_2", "center", "component"]], 1)
            transfo, R, t = rigid_sitk(ar_points_us, ar_points_mr)

            listToStr = "-".join([str(elem) for elem in modalities_set])
            flnm = f"{case}-{us}-{listToStr}"
            sitk.WriteTransform(transfo, path_output.format(annotator, NB_cases, seed, case, "reg", flnm + ".tfm"))

            # Resample MR modalities into US space
            us_sitk = sitk.ReadImage(path_us_all.format(us))
            for mod in modalities_set:
                mr_to_register = [k[1] for k in images_modalities if k[0] == mod][0]
                mr_to_register_flnm = os.path.basename(mr_to_register).replace(".nrrd", "")

                # MR->ref MR transform (if needed)
                if mr_flnm_or in mr_to_register:
                    mr_transformation = sitk.Transform()
                else:
                    res_filnm = os.path.join(
                        output_path_reg_mr, case, f"{case}-{mr_flnm_or}-to-{mr_to_register_flnm}.tfm"
                    )
                    mr_transformation = sitk.ReadTransform(res_filnm).GetInverse()

                mr = sitk.ReadImage(mr_to_register)
                mr = zero_mean(mr)

                final_transform_mr = sitk.CompositeTransform([mr_transformation, transfo])
                mr_resample = sitk.Resample(mr, us_sitk, final_transform_mr, sitk.sitkLinear)
                mr_resample = quantisize(mr_resample, 256)
                mr_resample = mask(mr_resample, us_sitk)

                mr_resample_path = path_output.format(annotator, NB_cases, seed, case, "data", flnm + f"_{mod}.nii.gz")
                sitk.WriteImage(
                    mr_resample,
                    mr_resample_path,
                )
                
                reslice_img = create_reslicing(path_output_data, us_mask_probe, flnm + f"_{mod}.nii.gz")
                mr_resample_slice_path = path_output.format(annotator, NB_cases, seed, case, 'data_reslice', flnm + f"_{mod}.nii.gz")
                reslice_img.to_filename(mr_resample_slice_path)                           
                    

            # Resample segmentation into US space
            final_transform_seg = sitk.CompositeTransform([transform_label, transfo])
            seg_resample = resample_seg(seg, us_sitk, final_transform_seg, labels=[0, 1, 2, 4])
            seg_resample = mask(seg_resample, us_sitk)
            seg_resample_path = path_output.format(annotator, NB_cases, seed, case, "data", flnm + "_seg.nii.gz")
            sitk.WriteImage(
                seg_resample,
                seg_resample_path,
            )
            
            reslice_img = create_reslicing(path_output_data, us_mask_probe, flnm + "_seg.nii.gz")
            mr_resample_slice_path = path_output.format(annotator, NB_cases, seed, case, 'data_reslice', flnm + "_seg.nii.gz")
            reslice_img.to_filename(mr_resample_slice_path)                

            # Optionally save landmarks as a labeled NIfTI volume
            if save_landmark:
                out = np.zeros_like(data_us)
                all_points = []

                # US-space landmark voxels
                for k in ["corner_1", "corner_2", "center", "component"]:
                    p = points_us[k].reshape(-1).tolist() + [1]
                    all_points.append(np.linalg.inv(affine_us).dot(p)[:3])

                # MR landmarks transformed into US space via (R, t), then voxelized in US affine
                for k in ["corner_1", "corner_2", "center", "component"]:
                    p = (R @ points_mr[k].reshape(-1, 1) + t).reshape(-1).tolist() + [1]
                    all_points.append(np.linalg.inv(affine_us).dot(p)[:3])

                for i, p in enumerate(all_points):
                    temp_i = np.zeros_like(data_us)
                    temp_i[int(p[0]), int(p[1]), int(p[2])] = 1
                    temp_i = ndimage.binary_dilation(temp_i, iterations=2).astype(temp_i.dtype)
                    out += (i + 1) * temp_i

                nib.Nifti1Image(out, affine_us).to_filename(
                    path_output.format(annotator, NB_cases, seed, case, "landmarks", flnm + "_landmarks.nii.gz")
                )

            count += 1

        except Exception as e:
            print(e, case, us)

# --------------------
# 5) Generate ground-truth label in the case's own US space
# --------------------
path_us_ref = path_us_all.format(case)
img_us_ref = sitk.ReadImage(path_us_ref)

res_filnm = os.path.join(
    output_path_reg_us,
    case,
    f"{case}-{case}-intraop-US-pre_dura-to-{mr_flnm_or}.tfm",
)
transform_tous = sitk.ReadTransform(res_filnm).GetInverse()

final_transform_gt = sitk.CompositeTransform([transform_label, transform_tous])
gt_resample = resample_seg(seg, img_us_ref, final_transform_gt, labels=[0, 1, 2, 4])
gt_resample = mask(gt_resample, img_us_ref)
sitk.WriteImage(gt_resample, path_output.format(annotator, NB_cases, seed, case, "label", "gt_seg.nii.gz"))

# --------------------
# 6) Generate "tracking" label 
# --------------------
final_transform_gt = sitk.CompositeTransform([transform_label])
gt_resample = resample_seg(seg, img_us_ref, final_transform_gt, labels=[0, 1, 2, 4])
gt_resample = mask(gt_resample, img_us_ref)
sitk.WriteImage(gt_resample, path_output.format(annotator, NB_cases, seed, case, "tracking", f"{case}.nii.gz"))



    
                
