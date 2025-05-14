### Individual naming/format
# NODDI: [32, 35, ..., 49, 50]
# Oddball: [sub001, sub002, ..., sub016, sub017]
# CNEPFL: [sub-02, sub-04, ..., sub-24, sub-26]

## To train E2fNet
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 32 --fmri_channel 30 --exp_root ./run/NODDI-32 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 35 --fmri_channel 30 --exp_root ./run/NODDI-35 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 36 --fmri_channel 30 --exp_root ./run/NODDI-36 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 37 --fmri_channel 30 --exp_root ./run/NODDI-37 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 38 --fmri_channel 30 --exp_root ./run/NODDI-38 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 39 --fmri_channel 30 --exp_root ./run/NODDI-39 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 40 --fmri_channel 30 --exp_root ./run/NODDI-40 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 42 --fmri_channel 30 --exp_root ./run/NODDI-42 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 43 --fmri_channel 30 --exp_root ./run/NODDI-43 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 44 --fmri_channel 30 --exp_root ./run/NODDI-44 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 45 --fmri_channel 30 --exp_root ./run/NODDI-45 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 46 --fmri_channel 30 --exp_root ./run/NODDI-46 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 47 --fmri_channel 30 --exp_root ./run/NODDI-47 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 48 --fmri_channel 30 --exp_root ./run/NODDI-48 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 49 --fmri_channel 30 --exp_root ./run/NODDI-49 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset NODDI --test_ids 50 --fmri_channel 30 --exp_root ./run/NODDI-50 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub001 --fmri_channel 32 --exp_root ./run/Oddball-sub001 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub002 --fmri_channel 32 --exp_root ./run/Oddball-sub002 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub003 --fmri_channel 32 --exp_root ./run/Oddball-sub003 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub004 --fmri_channel 32 --exp_root ./run/Oddball-sub004 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub005 --fmri_channel 32 --exp_root ./run/Oddball-sub005 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub006 --fmri_channel 32 --exp_root ./run/Oddball-sub006 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub007 --fmri_channel 32 --exp_root ./run/Oddball-sub007 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub008 --fmri_channel 32 --exp_root ./run/Oddball-sub008 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub009 --fmri_channel 32 --exp_root ./run/Oddball-sub009 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub010 --fmri_channel 32 --exp_root ./run/Oddball-sub010 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub011 --fmri_channel 32 --exp_root ./run/Oddball-sub011 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub012 --fmri_channel 32 --exp_root ./run/Oddball-sub012 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub013 --fmri_channel 32 --exp_root ./run/Oddball-sub013 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub014 --fmri_channel 32 --exp_root ./run/Oddball-sub014 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub015 --fmri_channel 32 --exp_root ./run/Oddball-sub015 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub016 --fmri_channel 32 --exp_root ./run/Oddball-sub016 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset Oddball --test_ids sub017 --fmri_channel 32 --exp_root ./run/Oddball-sub017 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2


CUDA_VISIBLE_DEVICES=0 python3 train_Spectrogram2fMRI.py --dataset CNEPFL --test_ids sub-21 sub-22 sub-24 sub-26 --fmri_channel 30 --exp_root ./run/CNEPFL --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2

# ## To train E2fGAN
# CUDA_VISIBLE_DEVICES=0 python train_E2fGAN.py \
# --dataset NODDI \
# --test_ids 43 \
# --fmri_channel 30 \
# --exp_root /home/Experiments/EEG2fMRI \
# --batch_size 32 \
# --num_epochs 50 \
# --lr 0.001

CUDA_VISIBLE_DEVICES=0 python3 train_E2fNet.py --dataset NODDI --test_ids 49 --fmri_channel 30 --exp_root ./run/NODDI-49 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_E2fNet.py --dataset Oddball --test_ids sub016 --fmri_channel 32 --exp_root ./run/Oddball-sub016 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_E2fNet.py --dataset CNEPFL --test_ids sub-21 sub-22 sub-24 sub-26 --fmri_channel 30 --exp_root ./run/CNEPFL --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES=0 python3 train_E2fGAN.py --dataset NODDI --test_ids 49 --fmri_channel 30 --exp_root ./run/NODDI-49 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_E2fGAN.py --dataset Oddball --test_ids sub016 --fmri_channel 32 --exp_root ./run/Oddball-sub016 --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2
CUDA_VISIBLE_DEVICES=0 python3 train_E2fGAN.py --dataset CNEPFL --test_ids sub-21 sub-22 sub-24 sub-26 --fmri_channel 30 --exp_root ./run/CNEPFL --batch_size 16 --num_epochs 50 --lr 0.001 --gradient_accumulation_steps 2