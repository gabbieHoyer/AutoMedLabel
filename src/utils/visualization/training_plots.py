
import os
import json
import gc
import logging
import matplotlib.pyplot as plt

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

logger = logging.getLogger(__name__)

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