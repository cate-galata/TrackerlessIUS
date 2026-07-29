import torch
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from medpy.metric import dc, assd, hd95, precision, recall
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from monai.metrics import compute_surface_dice
import argparse


def binary_relative_volume_error(g_volume, s_volume):
    s_v = s_volume.sum()
    g_v = g_volume.sum()
    assert(g_v > 0)
    rve = abs(s_v - g_v)/max([float(g_v),float(s_v)])
    return rve

def dc_mine(g_volume, s_volume):
    if np.sum(g_volume)>0:
        return 100*dc(g_volume,s_volume)
    else:
        if np.sum(s_volume)>0:
            return 0
        else:
            return 100

parser = argparse.ArgumentParser(
                    prog='Metrics',
                    description='This script computes the metrics')

parser.add_argument('--case', type=str, default="Case112")

args = parser.parse_args()

# --------------------
# Hyperparameters
# --------------------
cases = [args.case]
path_predictions = './experiments/test_set/'

cases_n1 = ['Case011', 'Case025','Case045','Case052', 'Case056', 'Case070', 'Case085', 'Case103', 'Case112','Case114']
cases_n2 = ['Case027', 'Case045', 'Case074', 'Case085', 'Case099', 'Case103', 'Case112']

print(f'### VOLUMETRIC ###')

all_scores = dict() 
protocols_eval = ['remind', 'erikson']
methods_eval = ['resect', 'brats', 'bratious', 'tracking', '10_100_2_miccai', 'otf_100_2',]
name = {'resect': 'ReSECT Unet','brats':'Brats Unet', 'bratious': 'Bratious Unet', 'tracking': 'Navigation', '10_100_2_miccai': 'K=10 MICCAI', 'otf_100_2': 'OTF Unet'}

for method in methods_eval:
    all_scores[method] = dict()
    string = f'{name[method]} &'
    for protocol in protocols_eval:
        if protocol == 'remind':
            cases = cases_n1
        else:
            cases = cases_n2
        all_scores[method][protocol] = dict()
        path_folder = os.path.join(path_predictions, protocol, method)
        path_folder_gt = os.path.join(path_predictions, protocol, 'gt')
        scores = {'dice':[], 'assd':[], 'precision': [], 'recall': []}
        for case in cases:
            gt = sitk.ReadImage(os.path.join(path_folder_gt, f'reslice{case}_crop.nii.gz'))
            pred = sitk.ReadImage(os.path.join(path_folder, f'reslice{case}_crop.nii.gz'))
            spacing = [0.5, 0.5, 0.5]
            
            gt = sitk.Resample(gt, pred, sitk.Transform(), sitk.sitkNearestNeighbor)
            
            gt_data =  (sitk.GetArrayFromImage(gt).transpose().squeeze()>0).astype(np.int16)
            
            pred_data =  (sitk.GetArrayFromImage(pred).transpose().squeeze()>0).astype(np.int16)

            scores['dice'].append(100*dc(pred_data, gt_data))
            scores['assd'].append(assd(pred_data, gt_data, voxelspacing=spacing))

            scores['precision'].append(100*precision(pred_data, gt_data))
            scores['recall'].append(100*recall(pred_data, gt_data))

        all_scores[method][protocol] = scores
        for metric in ['dice', 'assd', 'precision', 'recall']:
            string = string + f"{np.median(scores[metric]):.1f} ({np.quantile(scores[metric],0.75)-np.quantile(scores[metric],0.25):.1f}) &"
    print(string[:-1] + '\\\\')

print('### SLICE-BY-SLICE ###')

all_scores = dict() 
protocols_eval = ['slice']
methods_eval = ['resect', 'brats', 'bratious', 'tracking', '10_100_2_miccai', 'otf_100_2', 'manual']
name = {'resect': 'ReSECT Unet','brats':'Brats Unet', 'bratious': 'Bratious', 'tracking': 'Navigation', '10_100_2_miccai': 'K=10 MICCAI', 'otf_100_2': 'OTF Unet', 'manual': 'Manual'}

for method in methods_eval:
    all_scores[method] = dict()
    string = f'{name[method]} &'
    for protocol in protocols_eval:
        all_scores[method][protocol] = dict()
        path_folder = os.path.join(path_predictions, protocol, method)
        path_folder_gt = os.path.join(path_predictions, protocol, 'gt')
        scores = {'dice':[], 'assd':[], 'rve':[]}
        for case in cases_n2:
            gt = sitk.ReadImage(os.path.join(path_folder_gt, f'reslice{case}_crop.nii.gz'))
            pred = sitk.ReadImage(os.path.join(path_folder, f'reslice{case}_crop.nii.gz'))
            spacing = [0.5, 0.5]
            
            gt = sitk.Resample(gt, pred, sitk.Transform(), sitk.sitkNearestNeighbor)
            
            gt_data =  (sitk.GetArrayFromImage(gt).transpose().squeeze()>0).astype(np.int16)
            
            pred_data =  (sitk.GetArrayFromImage(pred).transpose().squeeze()>0).astype(np.int16)
            for z in range(pred_data.shape[-1]):
                gt_data_slice = gt_data[...,z]
                pred_data_slice = pred_data[...,z]
                scores['dice'].append(dc_mine(gt_data_slice,pred_data_slice))
                if gt_data_slice.sum()>0:
                    scores['rve'].append(binary_relative_volume_error(gt_data_slice,pred_data_slice))
                    if pred_data_slice.sum()>0:
                        scores['assd'].append(assd(gt_data_slice,pred_data_slice,voxelspacing=spacing))
                    else:
                        scores['assd'].append(10)
                else:
                    if pred_data_slice.sum()>0:
                        scores['rve'].append(1)
                        scores['assd'].append(10)
                    else:
                        scores['rve'].append(0)
                        scores['assd'].append(0)
        print(5*'--' + f' {method} ' + 5*'--')
        all_scores[method][protocol] = scores
        for metric in ['dice', 'assd']:
            string = string + f"{np.median(scores[metric]):.1f} ({np.quantile(scores[metric],0.75)-np.quantile(scores[metric],0.25):.1f}) &"
    print(string[:-1] + '\\\\')
