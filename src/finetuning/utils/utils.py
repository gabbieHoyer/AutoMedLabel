
import os
import matplotlib.pyplot as plt
import wandb
import json
import os
import json
import gc

import torch.distributed as dist

# Note: because these are functions particular to Finetuning, they should be
# kept in finetuning directory rather than moved to src/utils

# ------- FUNCTIONS TO DETERMINE EXPERIMENT RUN OUTPUT DIRECTORY ------- #
def determine_run_directory(base_dir, task_name, group_name=None):
    """
    Determines the next run directory for storing experiment data.
    """
    if group_name !=None:
        base_path = os.path.join(base_dir, task_name, group_name)
    else:
        base_path = os.path.join(base_dir, task_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Filter for directories that start with 'Run_' and are followed by an integer
    existing_runs = []
    for d in os.listdir(base_path):
        if d.startswith('Run_') and os.path.isdir(os.path.join(base_path, d)):
            parts = d.split('_')
            if len(parts) == 2 and parts[1].isdigit():  # Check if there is a number after 'Run_'
                existing_runs.append(d)
    
    if existing_runs:
        # Sort by the integer value of the part after 'Run_'
        existing_runs.sort(key=lambda x: int(x.split('_')[-1]))
        last_run_num = int(existing_runs[-1].split('_')[-1])
        next_run_num = last_run_num + 1
    else:
        next_run_num = 1
    
    run_directory = f'Run_{next_run_num}'
    full_run_path = os.path.join(base_path, run_directory)
    os.makedirs(full_run_path, exist_ok=True)
    
    return full_run_path

# ------- FUNCTIONS FOR LOSS SAVING AND TRAIN/VAL CURVES ------- #

def plot_losses(losses, model_save_path, run_id, mode):
    plt.figure(figsize=(10, 5))

    epochs = [entry['epoch'] for entry in losses]  # Extract epochs
    loss_values = [entry['loss'] for entry in losses]  # Extract loss values
    
    plt.plot(epochs, loss_values, label=f'{mode.capitalize()} Loss')
    plt.title(f"{mode.capitalize()} BCE + DICE Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = os.path.join(model_save_path, 'train_val_curves')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"{run_id}_{mode}_loss_over_epochs.png"))
    plt.close()
    gc.collect()  # Optionally clear memory after plotting


def plot_combined_losses(train_losses, val_losses, model_save_path, run_id):
    plt.figure(figsize=(10, 5))
    
    # Extract data
    train_epochs = [entry['epoch'] for entry in train_losses]
    train_loss_values = [entry['loss'] for entry in train_losses]
    val_epochs = [entry['epoch'] for entry in val_losses]
    val_loss_values = [entry['loss'] for entry in val_losses]
    
    plt.plot(train_epochs, train_loss_values, label='Training Loss')
    plt.plot(val_epochs, val_loss_values, label='Validation Loss')
    
    plt.title("Training and Validation BCE + DICE Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Ensure the directory exists
    plot_path = os.path.join(model_save_path, 'train_val_curves')
    os.makedirs(plot_path, exist_ok=True)
    
    plt.savefig(os.path.join(plot_path, f"{run_id}_combined_loss_over_epochs.png"))
    plt.close()

def plot_metrics(metrics, model_save_path, run_id, metric_name="dice_score", mode="val"):
    plt.figure(figsize=(10, 5))

    epochs = [entry['epoch'] for entry in metrics]  # Extract epochs
    
    metric_values = [entry[metric_name.lower()] for entry in metrics]  # Extract metric values

    plt.plot(epochs, metric_values, label=f'{mode.capitalize()} {metric_name}')
    plt.title(f"{mode.capitalize()} {metric_name} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plot_path = os.path.join(model_save_path, 'metrics_curves')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"{run_id}_{mode}_{metric_name.lower()}_over_epochs.png"))
    plt.close()
    gc.collect()  # Optionally clear memory after plotting


def save_losses(shared_losses, model_save_path, run_id):
    # Ensure the directory exists for the JSON file
    losses_file_path = os.path.join(model_save_path, 'train_val_curves', f"{run_id}_losses.json")
    os.makedirs(os.path.dirname(losses_file_path), exist_ok=True)

    # Write the shared_losses dictionary to a JSON file
    with open(losses_file_path, 'w') as f:
        json.dump(shared_losses, f, indent=4)  # Use indent for pretty printing

    print(f"Losses saved to {losses_file_path}")

def save_metrics(shared_metrics, model_save_path, run_id):
    # Ensure the directory exists for the JSON file
    metrics_file_path = os.path.join(model_save_path, 'metrics_curves', f"{run_id}_metrics.json")
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

    # Write the shared_metrics dictionary to a JSON file
    with open(metrics_file_path, 'w') as f:
        json.dump(shared_metrics, f, indent=4)  # Use indent for pretty printing

    print(f"Metrics saved to {metrics_file_path}")



# -------- MISCELLANEOUS TOOLS -------- #
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt



# def determine_run_directory(base_dir, task_name, group_name):
#     """
#     Determines the next run directory for storing experiment data.
#     """
#     base_path = os.path.join(base_dir, task_name, group_name)
#     os.makedirs(base_path, exist_ok=True)
    
#     existing_runs = [d for d in os.listdir(base_path) if d.startswith('Run_') and os.path.isdir(os.path.join(base_path, d))]
#     if existing_runs:
#         existing_runs.sort()
#         last_run_num = int(existing_runs[-1].split('_')[-1])
#         next_run_num = last_run_num + 1
#     else:
#         next_run_num = 1
    
#     run_directory = f'Run_{next_run_num}'
#     full_run_path = os.path.join(base_path, run_directory)
#     os.makedirs(full_run_path, exist_ok=True)
    
#     return full_run_path

# def plot_losses(losses, model_save_path, run_id, mode):
#     plt.figure(figsize=(10, 5))
#     # Extract loss values and epochs from the list of dictionaries
#     epochs = [entry['epoch'] for entry in losses]
#     loss_values = [entry['loss'] for entry in losses]
    
#     plt.plot(epochs, loss_values, label=f'{mode.capitalize()} Loss')
#     plt.title(f"{mode.capitalize()} Loss Over Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
    
#     # Ensure the directory exists
#     plot_path = os.path.join(model_save_path, 'train_val_curves')
#     os.makedirs(plot_path, exist_ok=True)
    
#     plt.savefig(os.path.join(plot_path, f"{run_id}_{mode}_loss_over_epochs.png"))
#     plt.close()


# def plot_combined_losses(train_losses, val_losses, model_save_path, run_id):
#     plt.figure(figsize=(10, 5))
    
#     # Plot training and validation losses
#     epochs = range(1, len(train_losses) + 1)
#     plt.plot(epochs, train_losses, label='Training Loss')
#     plt.plot(epochs, val_losses, label='Validation Loss')
    
#     # Add titles and labels
#     plt.title("Training and Validation Loss Over Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
    
#     # Save the figure
#     plt.savefig(os.path.join(model_save_path, 'train_val_curves', f"{run_id}_combined_loss_over_epochs.png"))
#     plt.close()


# def plot_losses(losses, model_save_path, run_id, mode):
#     plt.figure(figsize=(10, 5))
#     plt.plot(losses, label=f'{mode.capitalize()} Loss')
#     plt.title(f"{mode.capitalize()} Dice + Cross Entropy Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig(os.path.join(model_save_path, 'train_val_curves', f"{run_id}_{mode}_loss_over_epochs.png"))
#     plt.close()


# def save_losses(shared_losses, model_save_path, run_id):
#     # Construct the file path
#     losses_file_path = os.path.join(model_save_path, 'train_val_curves', f"{run_id}_losses.json")

#     # Ensure the directory exists
#     os.makedirs(model_save_path, exist_ok=True)

#     # Write the shared_losses dictionary (with both train and val losses) to a JSON file
#     with open(losses_file_path, 'w') as f:
#         json.dump(shared_losses, f, indent=4)  # Use indent for pretty printing

#     print(f"Losses saved to {losses_file_path}")
