import os
import glob
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import torchmetrics

import diffusers.optimization as optimization
from accelerate import Accelerator

from eeg2fmri_datasets import EEG2fMRIDataset
import utils
from Spectrogram2fMRI import Spectrogram2fMRI
from losses import ssim_mse_loss
import data_cfg

import argparse
import json
from rich import print

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--n_worker", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--mixed_precision", default="no")
parser.add_argument("--lr_warmup_steps", type=int, default=50)
parser.add_argument("--save_every", type=int, default=10, help="Save model every epochs")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exp_root", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True, choices=["NODDI", "Oddball", "CNEPFL"])
parser.add_argument("--test_ids", nargs="+", type=str, required=True)
parser.add_argument("--fmri_channel", type=int, required=True, help="Target fMRI channel")


### TRAIN LOOP
def train_loop(config, model, optimizer, train_dataloader,
               test_dataloader, lr_scheduler, loss_func):
    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # create exp dir
    current_timestr = utils.get_current_timestr()
    exp_dir = Path(config.exp_root) / f"Spectrogram2fMRI_{config.dataset}_{current_timestr}"
    if accelerator.is_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(exp_dir / 'visual_res', exist_ok=True)
        print(f'Exp dir: {exp_dir}')
        # save config
        with open(exp_dir / 'commandline_args.txt', 'w') as f:
            json.dump(config.__dict__, f, indent=2)

        accelerator.init_trackers("train_example")

    # prepare everything
    # there is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler, loss_func, ssim_metric = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler, loss_func, ssim_metric
    )

    global_step = 0
    progress_bar = tqdm(total=config.num_epochs, disable=not accelerator.is_local_main_process)
    best_ssim = -1.0
    curr_ssim = -1.0

    for epoch in range(config.num_epochs):
        model.train()
        total_batch = len(train_dataloader)

        for idx, batch in enumerate(train_dataloader):
            eeg_batch, fmri_batch = batch

            with accelerator.accumulate(model):
                pred_fmri = model(eeg_batch)
                loss = loss_func(pred_fmri, fmri_batch)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch} [{idx}/{total_batch}]")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "best_ssim": best_ssim,
                "curr_ssim": curr_ssim,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            model.eval()
            total_batch = len(test_dataloader)
            test_ssim_score = 0.0

            for idx, batch in enumerate(test_dataloader):
                eeg_batch, fmri_batch = batch
                # model pred
                with torch.no_grad():
                    pred_fmri = model(eeg_batch)

                # update progress bar
                progress_bar.set_description(f"Testing [{idx}/{total_batch}]")

                # calculate SSIM
                test_ssim_score += ssim_metric(pred_fmri, fmri_batch).item()

            curr_ssim = test_ssim_score / total_batch
            logs["curr_ssim"] = curr_ssim

            # save latest visual result
            slice_idx = 16  # save a slice out of fMRI volume
            pred_slice = pred_fmri[0, slice_idx].cpu().numpy() * 255.
            img = Image.fromarray(pred_slice).convert('RGB')
            img.save(f'{exp_dir}/visual_res/latest_result.png')

            # save latest model
            torch.save(model.state_dict(), f"{exp_dir}/latest.pth")

            # save best model and visual result
            if curr_ssim > best_ssim:
                best_ssim = curr_ssim
                logs["best_ssim"] = best_ssim

                # save image
                img = Image.fromarray(pred_slice).convert('RGB')
                img.save(f'{exp_dir}/visual_res/epoch_{epoch}_SSIM_{curr_ssim:.5f}.png')

                # remove the previous best model
                if len(glob.glob(f"{exp_dir}/*best_SSIM*.pth")):
                    best_path = glob.glob(f"{exp_dir}/*best_SSIM*.pth")[0]
                    os.remove(best_path)

                torch.save(model.state_dict(), f"{exp_dir}/epoch_{epoch}_best_SSIM_{curr_ssim:.5f}.pth")

            # save model at every n epochs
            if (epoch + 1) % config.save_every == 0:
                torch.save(model.state_dict(), f"{exp_dir}/epoch_{epoch}_SSIM_{curr_ssim:.5f}.pth")

            # update progress bar
            progress_bar.set_postfix(**logs)

        progress_bar.update(1)


if __name__ == "__main__":
    # create config
    config = parser.parse_args()
    print(config)

    # get individual list
    data_root = Path(data_cfg.processed_data_roots[config.dataset])

    print(data_root,os.listdir(data_root))

    if config.dataset == "NODDI":
        data_list = [
            Path(indv).stem for indv in os.listdir(data_root) if (Path(data_root) / indv).is_file()
        ]
    else:
        data_list = [x for x in os.listdir(data_root) if os.path.isdir(data_root / x)]

    # collect train and test lists
    assert set(config.test_ids).issubset(set(data_list)), f"{config.test_ids} not in {sorted(data_list)}"
    test_list = config.test_ids
    train_list = list(set(data_list) - set(config.test_ids))

    # get individual data
    idv_train = sorted(utils.get_individual_list(config.dataset, train_list))
    idv_test = sorted(utils.get_individual_list(config.dataset, test_list))

    print(f"Data root: {data_root}")

    print(f"Train individuals: {train_list}")
    print(f"Test individuals: {test_list}")

    print(f"Train data: {idv_train}")
    print(f"Test data: {idv_test}")

    print("Loading data ...")
    eeg_train, fmri_train = utils.load_h5_from_list(data_root, idv_train)
    eeg_test, fmri_test = utils.load_h5_from_list(data_root, idv_test)
    print(f"Train EEG shape: {eeg_train.shape} | Train fMRI shape: {fmri_train.shape}")
    print(f"Test EEG shape: {eeg_test.shape} | Test fMRI shape: {fmri_test.shape}")

    # get min max range
    min_eeg = float(min(eeg_train.min(), eeg_test.min()))
    max_eeg = float(max(eeg_train.max(), eeg_test.max()))
    print(f"EEG min-max range: {(min_eeg, max_eeg)}")

    eeg_train = utils.normalize_data(eeg_train, base_range=(min_eeg, max_eeg))
    eeg_test = utils.normalize_data(eeg_test, base_range=(min_eeg, max_eeg))

    # save to config
    config.train_list = train_list
    config.test_list = test_list
    config.idv_train = idv_train
    config.idv_test = idv_test

    config.min_eeg = min_eeg
    config.max_eeg = max_eeg

    # create datasets
    train_dataset = EEG2fMRIDataset(eeg_train, fmri_train)
    test_dataset = EEG2fMRIDataset(eeg_test, fmri_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              num_workers=config.n_worker, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             num_workers=config.n_worker, shuffle=False)

    # define networks
    model =  Spectrogram2fMRI(in_dim=20,hidden_dim=256,out_dim=config.fmri_channel)


    ### prepare for training
    loss_func = ssim_mse_loss(lambda_ssim=0.5, lambda_mes=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
        num_cycles=max(1, config.num_epochs // 10),  # restart every 10 epochs
    )
    # initialize accelerator and logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # run training
    train_loop(config, model, optimizer, train_loader, test_loader, lr_scheduler, loss_func)