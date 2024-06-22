import os
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







# save_checkpoint({
#     "model": model.state_dict(),
#     "optimizer": optimizer.state_dict(),
#     "epoch": epoch,
#     "train_loss": train_loss,
#     "val_loss": val_loss
# }, latest_checkpoint_path)

# def save_checkpoint(state, filepath):
#     with open(filepath, 'wb') as f:
#         torch.save(state, f)
#         f.flush()  # Explicitly flush the file buffer
#     os.fsync(f.fileno())  # Ensure all internal buffers associated with the file are written to disk





# def log_and_checkpoint(mode, average_loss, module_cfg, model, optimizer, epoch, model_save_path, run_id, best_loss):
#     if not GPUSetup.is_main_process():
#         # Skip logging and checkpointing for non-primary processes in distributed training
#         return best_loss
    
#     # Logging the average loss to Weights & Biases, if applicable
#     if module_cfg['use_wandb']:
#         wandb.log({f"{mode}_epoch_loss": average_loss})

#     # Print epoch loss
#     print(f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, Mode: {mode.capitalize()}, Epoch Loss: {average_loss}", flush=True)

#     # Checkpointing logic for training mode
#     if mode == 'train':
#         # Construct checkpoint paths
#         latest_checkpoint_path = os.path.join(model_save_path, f"{run_id}_finetuned_model_latest.pth")
#         best_checkpoint_path = os.path.join(model_save_path, f"{run_id}_finetuned_model_best.pth")

#         # Always save the latest model
#         torch.save({
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": epoch,
#             "loss": average_loss,
#         }, latest_checkpoint_path)

#         # Save the best model if current loss is lower
#         if average_loss < best_loss:
#             best_loss = average_loss  # Update best loss
#             torch.save({
#                 "model": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "epoch": epoch,
#                 "loss": average_loss,
#             }, best_checkpoint_path)

#             # Optionally log best model checkpoint to W&B
#             if module_cfg['use_wandb']:
#                 wandb.save(best_checkpoint_path)

#     return 

# best_loss




