# Parameters to process the raw datasets
processing_cfg = {
    'bold_shift': 6,
    'eeg_limit': True,
    'eeg_f_limit': 250,
    'interval_eeg': 20,

    "NODDI": {
        'n_volumes': 294, # 300 - bold_shift
        'f_resample': 2.160,
    },

    "Oddball": {
        'n_volumes': 164, # 170 - bold_shift
        'f_resample': 2.0,
    },
    
    "CNEPFL": {
        'n_volumes': 364, # 370 - bold_shift
        'f_resample': 1.280,
    },
}

raw_data_roots = {
    "NODDI": "./data/NODDI/",
    "Oddball": "./data/Oddball/",
    "CNEPFL": "./data/CN-EPFL/ds002158-download/",
}

# Path to h5 data roots after datasets pre-processing. See docs/datasets_howto.md
processed_data_roots = {
    "NODDI": "./data/NODDI_h5_data",
    "Oddball": "./data/Oddball_h5_data",
    "CNEPFL": "./data/CNEPFL_h5_data",
}