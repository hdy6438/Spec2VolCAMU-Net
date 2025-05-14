import h5py
import numpy as np
from matplotlib import pyplot as plt

with h5py.File("data/CNEPFL_h5_data/sub-02_run-001.h5", 'r') as f:
    eeg_indv = np.array(f['eeg'][:])
    fmri_indv = np.array(f['fmri'][:])

sample_eeg = eeg_indv[0]
sample_fmri = fmri_indv[0]

print(f"EEG shape: {sample_eeg.shape}")
print(f"fMRI shape: {sample_fmri.shape}")

# for i in range(sample_eeg.shape[0]):
#     plt.figure(figsize=(8, 6))
#     plt.imshow(sample_eeg[i], cmap='jet', aspect='auto',interpolation='nearest')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(f"data_sample/eeg_{i}.png")
#     plt.close()
#
# for i in range(sample_fmri.shape[0]):
#     plt.figure(figsize=(8, 8))
#     plt.imshow(np.rot90(sample_fmri[i], k=-1), cmap='gray', aspect='auto',interpolation='nearest')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(f"data_sample/fmri_{i}.png")
#     plt.close()