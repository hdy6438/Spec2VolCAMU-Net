import os
import re

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import nibabel as nib
from nilearn import datasets, image, masking
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting

fmri_gt = np.stack([np.stack([cv2.imread(f"img/CNEPFL/{i}/gt_{j}.png", cv2.IMREAD_GRAYSCALE) for j in range(0, 30)], axis=0) for i in tqdm(range(0,350, 10))], axis=0)
fmri_gt_nii_data = nib.Nifti1Image(fmri_gt.transpose(1, 2, 3, 0) , np.eye(4))

fmri_E2fGAN = np.stack([np.stack([cv2.imread(f"img/CNEPFL/{i}/E2fGAN_{j}.png", cv2.IMREAD_GRAYSCALE) for j in range(0, 30)], axis=0) for i in tqdm(range(0,350, 10))], axis=0)
fmri_E2fGAN_nii_data = nib.Nifti1Image(fmri_E2fGAN.transpose(1, 2, 3, 0) , np.eye(4))

fmri_E2fNet = np.stack([np.stack([cv2.imread(f"img/CNEPFL/{i}/E2fNet_{j}.png", cv2.IMREAD_GRAYSCALE) for j in range(0, 30)], axis=0) for i in tqdm(range(0,350, 10))], axis=0)
fmri_E2fNet_nii_data = nib.Nifti1Image(fmri_E2fNet.transpose(1, 2, 3, 0) , np.eye(4))

fmri_Spectrogram2fMRI = np.stack([np.stack([cv2.imread(f"img/CNEPFL/{i}/Spectrogram2fMRI_{j}.png", cv2.IMREAD_GRAYSCALE) for j in range(0, 30)], axis=0) for i in tqdm(range(0,350, 10))], axis=0)
fmri_Spectrogram2fMRI_nii_data = nib.Nifti1Image(fmri_Spectrogram2fMRI.transpose(1, 2, 3, 0) , np.eye(4))


atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_img = image.load_img(atlas.filename)
atlas_res = image.resample_to_img(atlas_img, fmri_gt_nii_data, interpolation="nearest")
atlas_data = atlas_res.get_fdata().astype(int)
present_labels = np.unique(atlas_data)
present_labels = present_labels[present_labels > 0]
print("保留下来的 ROI index:", present_labels)
labels = np.array(atlas.labels)[present_labels]
masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True, t_r=2.0)

fmri_gt_time_series = masker.fit_transform(fmri_gt_nii_data)  # shape (T, n_ROIs)
fmri_E2fGAN_time_series = masker.fit_transform(fmri_E2fGAN_nii_data)  # shape (T, n_ROIs)
fmri_E2fNet_time_series = masker.fit_transform(fmri_E2fNet_nii_data)  # shape (T, n_ROIs)
fmri_Spectrogram2fMRI_time_series = masker.fit_transform(fmri_Spectrogram2fMRI_nii_data)  # shape (T, n_ROIs)

print(fmri_gt_time_series.shape)
print(fmri_E2fGAN_time_series.shape)
print(fmri_E2fNet_time_series.shape)
print(fmri_Spectrogram2fMRI_time_series.shape)



def clean_text(text):
    # 去掉括号及其中内容（支持中英文括号）
    text = re.sub(r'（.*?）|\(.*?\)', '', text)
    # 去掉第一个逗号及其后面的内容（支持中英文逗号）
    text = re.split(r'[，,]', text)[0]
    return text.strip()


fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i % 3 == 0:
        ax.set_ylabel('Signal Intensity')
    if i >= 3:
        ax.set_xlabel('Time (scans)')

    if i >=3:
        i+=1
    ax.plot(fmri_gt_time_series[:, i], label='Ground Truth')
    ax.plot(fmri_E2fGAN_time_series[:, i], label='E2fGAN')
    ax.plot(fmri_E2fNet_time_series[:, i], label='E2fNet')
    ax.plot(fmri_Spectrogram2fMRI_time_series[:, i], label='Spectrogram2fMRI')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(clean_text(labels[i]))


plt.tight_layout()
plt.savefig("test.png")

plt.show()

