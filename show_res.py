import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import data_cfg
import utils
from Spectrogram2fMRI import Spectrogram2fMRI
from eeg2fmri_datasets import EEG2fMRIDataset
from models import EEGEncoder, create_unet, fMRIDecoder, EEG2fMRINet

data_name = 'CNEPFL'

assert data_name in ['NODDI', 'Oddball', 'CNEPFL']

data_root = Path(data_cfg.processed_data_roots[data_name])

fmri_channel = 30
# NODDI
if data_name == 'NODDI':
    test_list = ['49']

    # global min-max value
    eeg_min = -3.904906883760493
    eeg_max = 7.937204954155734
    
# Oddball
elif data_name == 'Oddball':
    test_ID = [16] # test [9 - 10]
    test_list = []
    for idx in test_ID:
        indv_data = f"sub{idx:03}/task001_run001"
        test_list.append(indv_data)
    
    fmri_channel = 32
    
    # global min-max value
    eeg_min = -2.466110737041575
    eeg_max = 6.480417369333849
    
# CNEPFL
elif data_name == 'CNEPFL':
    individuals = sorted([Path(x).stem for x in os.listdir(data_root)])
    test_list = individuals[-4:]
    test_list = [f"{test}/{test}_run-001" for test in test_list]

    # global min-max value
    eeg_min = -4.551622643288133
    eeg_max = 7.93715188090758

eeg_test, fmri_test = utils.load_h5_from_list(data_root, test_list)

# normalize data (fmri_test is already in range [0 - 1])
eeg_test = utils.normalize_data(eeg_test, base_range=(eeg_min, eeg_max))

# create datasets
test_dataset = EEG2fMRIDataset(eeg_test, fmri_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = "cuda:0"

# ### Load models
Spectrogram2fMRI_model = Spectrogram2fMRI(in_dim=20, hidden_dim=256, out_dim=fmri_channel)
if data_name == 'CNEPFL':
    ckpt_path = list(Path(f'run/{data_name}').glob(f'Spectrogram2fMRI_{data_name}*/*best_SSIM*.pth'))[0]
elif data_name == 'NODDI':
    ckpt_path = list(Path(f'run/{data_name}/{data_name}-{test_list[0]}').glob(f'Spectrogram2fMRI_{data_name}*/*best_SSIM*.pth'))[0]
elif data_name == 'Oddball':
    ckpt_path = list(Path(f'run/{data_name}/{data_name}-sub{test_ID[0]:03}').glob(f'Spectrogram2fMRI_{data_name}*/*best_SSIM*.pth'))[0]
Spectrogram2fMRI_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
Spectrogram2fMRI_model = Spectrogram2fMRI_model.to(device)
Spectrogram2fMRI_model = Spectrogram2fMRI_model.eval()
print("Spectrogram2fMRI model loaded from: ", ckpt_path)


E2fNet_model = EEG2fMRINet(
    eeg_encoder=EEGEncoder(in_channels=20, img_size=64),
    unet_module=create_unet(in_channels=256, out_channels=256),
    fmri_decoder=fMRIDecoder(in_channels=256, out_channels=fmri_channel)
)
if data_name == 'CNEPFL':
    ckpt_path = list(Path(f'run/{data_name}').glob(f'E2fNet_{data_name}*/*best_SSIM*.pth'))[0]
elif data_name == 'NODDI':
    ckpt_path = list(Path(f'run/{data_name}/{data_name}-{test_list[0]}').glob(f'E2fNet_{data_name}*/*best_SSIM*.pth'))[0]
elif data_name == 'Oddball':
    ckpt_path = list(Path(f'run/{data_name}/{data_name}-sub{test_ID[0]:03}').glob(f'E2fNet_{data_name}*/*best_SSIM*.pth'))[0]
E2fNet_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
E2fNet_model = E2fNet_model.to(device)
E2fNet_model = E2fNet_model.eval()
print("E2fNet model loaded from: ", ckpt_path)


E2fGAN_model = EEG2fMRINet(
    eeg_encoder=EEGEncoder(in_channels=20, img_size=64),
    unet_module=create_unet(in_channels=256, out_channels=256),
    fmri_decoder=fMRIDecoder(in_channels=256, out_channels=fmri_channel)
)
if data_name == 'CNEPFL':
    ckpt_path = list(Path(f'run/{data_name}').glob(f'E2fGAN_{data_name}*/*best_SSIM*.pth'))[0]
elif data_name == 'NODDI':
    ckpt_path = list(Path(f'run/{data_name}/{data_name}-{test_list[0]}').glob(f'E2fGAN_{data_name}*/*best_SSIM*.pth'))[0]
elif data_name == 'Oddball':
    ckpt_path = list(Path(f'run/{data_name}/{data_name}-sub{test_ID[0]:03}').glob(f'E2fGAN_{data_name}*/*best_SSIM*.pth'))[0]
E2fGAN_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
E2fGAN_model = E2fGAN_model.to(device)
E2fGAN_model = E2fGAN_model.eval()
print("E2fGAN model loaded from: ", ckpt_path)

def save_img(f_data_name, f_sample_idx, f_fmri_batch, f_Spectrogram2fMRI_pred_fmri, f_E2fNet_pred_fmri, f_E2fGAN_pred_fmri, f_fmri_channel):
    os.makedirs(f"img/{f_data_name}/{f_sample_idx}", exist_ok=True)

    f_fmri_batch = np.transpose(f_fmri_batch, (1, 2, 0))
    f_Spectrogram2fMRI_pred_fmri = np.transpose(f_Spectrogram2fMRI_pred_fmri, (1, 2, 0))
    f_E2fNet_pred_fmri = np.transpose(f_E2fNet_pred_fmri, (1, 2, 0))
    f_E2fGAN_pred_fmri = np.transpose(f_E2fGAN_pred_fmri, (1, 2, 0))

    for d_idx in range(f_fmri_channel):
        plt.imsave(f"img/{f_data_name}/{f_sample_idx}/gt_{d_idx}.png", f_fmri_batch[:, :, d_idx], cmap='gray')
        plt.imsave(f"img/{f_data_name}/{f_sample_idx}/Spectrogram2fMRI_{d_idx}.png", f_Spectrogram2fMRI_pred_fmri[:, :, d_idx], cmap='gray')
        plt.imsave(f"img/{f_data_name}/{f_sample_idx}/E2fNet_{d_idx}.png", f_E2fNet_pred_fmri[:, :, d_idx], cmap='gray')
        plt.imsave(f"img/{f_data_name}/{f_sample_idx}/E2fGAN_{d_idx}.png", f_E2fGAN_pred_fmri[:, :, d_idx], cmap='gray')

    print("Saved images for sample index: ", f_sample_idx)


pool = Pool(os.cpu_count())

sample_idx = 0
for eeg_batch, fmri_batch in test_loader:
    eeg_batch = eeg_batch.to(device)
    fmri_batch = fmri_batch.to(device)

    # model prediction
    with torch.no_grad():
        Spectrogram2fMRI_pred_fmri = Spectrogram2fMRI_model(eeg_batch)
        E2fNet_pred_fmri = E2fNet_model(eeg_batch)
        E2fGAN_pred_fmri = E2fGAN_model(eeg_batch)


    Spectrogram2fMRI_pred_fmri = Spectrogram2fMRI_pred_fmri.squeeze().detach().cpu().numpy()
    E2fNet_pred_fmri = E2fNet_pred_fmri.squeeze().detach().cpu().numpy()
    E2fGAN_pred_fmri = E2fGAN_pred_fmri.squeeze().detach().cpu().numpy()

    fmri_batch = fmri_batch.squeeze().detach().cpu().numpy()

    if sample_idx % 10 == 0:
        pool.apply_async(save_img,args=(data_name, sample_idx, fmri_batch, Spectrogram2fMRI_pred_fmri, E2fNet_pred_fmri, E2fGAN_pred_fmri, fmri_channel))

    sample_idx += 1

# 等待所有进程完成
pool.close()
pool.join()
print("All images saved.")


