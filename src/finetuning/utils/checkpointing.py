import os
from collections import OrderedDict
from datetime import datetime
from os.path import join
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

# import src.finetuning.utils.gpu_setup as GPUSetup
from . import gpu_setup as GPUSetup
from src.finetuning.utils.logging import wandb_log

def log_and_checkpoint(mode, train_loss, val_loss, module_cfg, model, optimizer, epoch, model_save_path, run_id, best_train_loss, best_val_loss):
    if not GPUSetup.is_main_process():
        return best_train_loss, best_val_loss

    # Construct checkpoint paths
    latest_checkpoint_path = os.path.join(model_save_path, f"{run_id}_finetuned_model_latest_epoch_{epoch}.pth")
    best_checkpoint_path = os.path.join(model_save_path, f"{run_id}_finetuned_model_best.pth")

    # Log losses to Weights & Biases, if applicable
    if module_cfg.get('use_wandb', False):
        if mode == 'both':
            wandb_log({"train_epoch_loss": train_loss, "val_epoch_loss": val_loss})
        else:
            wandb_log({f"{mode}_epoch_loss": train_loss if mode == 'train' else val_loss})

    # Print epoch loss
    print(f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, Training Loss: {train_loss}, Validation Loss: {val_loss}", flush=True)

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, latest_checkpoint_path)

    # Update best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss  # Update best validation loss
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
        }, best_checkpoint_path)

    if train_loss < best_train_loss:
        best_train_loss = train_loss  # Update best training loss

        # Optionally log best model checkpoint to W&B
        # if module_cfg.get('use_wandb', False):
        #     wandb.save(best_checkpoint_path)

    return best_train_loss, best_val_loss


# Function to check if the 'module.' prefix is present
def check_module_prefix(state_dict):
    return any(k.startswith('module.') for k in state_dict.keys())

def final_checkpoint_conversion(module_cfg, model_save_path, run_id):

    best_checkpoint_path = os.path.join(model_save_path, f"{run_id}_finetuned_model_best.pth")
    converted_checkpoint_path = os.path.join(model_save_path, f"{run_id}_finetuned_model_best_converted.pth")

    # Load the fine-tuned checkpoint
    finetuned_ckpt = torch.load(best_checkpoint_path)
    
    # Correct the 'model' keys if the checkpoint was saved from a multi-GPU setup
    if 'model' in finetuned_ckpt and check_module_prefix(finetuned_ckpt['model']):
        new_model_state_dict = OrderedDict()
        for k, v in finetuned_ckpt['model'].items():
            new_key = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_model_state_dict[new_key] = v
        finetuned_ckpt['model'] = new_model_state_dict
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(converted_checkpoint_path), exist_ok=True)
    
    # Save the updated checkpoint
    torch.save(finetuned_ckpt, converted_checkpoint_path)

    # Log the converted checkpoint path to wandb
    if module_cfg.get('use_wandb', False):
        wandb_log({"final_checkpoint_path": converted_checkpoint_path})

