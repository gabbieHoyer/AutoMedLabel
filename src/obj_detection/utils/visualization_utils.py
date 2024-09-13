

# set_image_clim
# map_labels_to_colors
# add_colorbar
# visualize_full_pred
# visualize_input
# visualize_pred

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap

# ------------------- Visualization Tools ----------------------------------- #
def set_image_clim(image):
    image_norm = 'percentile'
    # can do percentiles here
    if image_norm == 'percentile':
        return [np.percentile(image[:], 2),np.percentile(image[:], 98)]
    else:
        return [np.min(image[:]),np.max(image[:])]

def map_labels_to_colors(pred_mask, mask_labels):
    # Define a colormap that can provide a distinct color for each class
    color_map = plt.get_cmap('rainbow', len(mask_labels) - 1)  # Exclude background

    # Create an empty RGBA image
    colored_mask = np.zeros((*pred_mask.shape, 4), dtype=np.float32)  # Initialize with zeros

    # Map each label to a consistent color from the colormap
    for label_value, label_name in mask_labels.items():
        if label_value == 0:  # Skip the background
            continue
        mask = (pred_mask == label_value)
        color = color_map((label_value - 1) / (len(mask_labels) - 1))  # Get consistent RGBA color
        colored_mask[mask] = color  # Apply color where the label matches

    return colored_mask


def add_colorbar(fig, ax, mask_labels):
    # Get unique labels excluding background
    unique_labels = [label for label in mask_labels if label != 0]

    # Create a colormap based on unique labels
    base_cmap = plt.get_cmap('rainbow', len(unique_labels))
    colors = base_cmap(np.linspace(0, 1, len(unique_labels)))
    new_cmap = ListedColormap(colors)

    # Normalize according to the number of unique labels
    norm = Normalize(vmin=min(unique_labels), vmax=max(unique_labels))

    # Create a scalar mappable for colormap and normalization
    sm = ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])  # dummy array

    # Add colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation='vertical', aspect=10)
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([mask_labels[label] for label in unique_labels])

def visualize_full_pred(image, pred_mask, mask_labels, image_name, model_save_path, image_clim=None):  
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with three subplots

    # Original image
    ax[0].imshow(image, cmap='gray', clim=image_clim)  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Prediction overlay
    if image_clim is not None:
        ax[1].imshow(image, cmap='gray', clim=image_clim)  # Original image
    else:
        ax[1].imshow(image, cmap='gray')  # Original image
    colored_pred_mask = map_labels_to_colors(pred_mask, mask_labels)
    im = ax[1].imshow(colored_pred_mask, alpha=0.5)  # Overlay the colored mask
    ax[1].set_title('Prediction Overlay')
    ax[1].axis('off')

    # Add colorbar
    add_colorbar(fig, ax, mask_labels)

    figure_file_path = os.path.join(model_save_path, "QC", f"{image_name}_full_pred.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory


# -----------------------------------------------------------------
def visualize_input(image, image_name, model_save_path):
    # Check the image shape and permute if necessary
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig = plt.figure(figsize=(10, 10))  # Create a figure with three subplots

    # Assuming image is in [C, H, W] format and is an RGB image
    plt.imshow(image, cmap='gray')  # Assuming image is in [C, H, W] format
    plt.title('Original Image')
    plt.axis('off')

    figure_file_path = os.path.join(model_save_path, "input_QC", f"{image_name}.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path) 
    plt.close(fig)  # Close the figure to free memory

# -----------------------------------------------------------------

def visualize_pred(image, pred_mask, binary_pred, boxes, image_name, model_save_path, image_clim=None):  # pred_mask_bin, label_id,
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with three subplots

    # Original image
    # Assuming image is in [C, H, W] format and is an RGB image
    ax[0].imshow(image, cmap='gray', clim=image_clim)  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Prediction overlay
    ax[1].imshow(image, cmap='gray', clim=image_clim)  # Original image
    ax[1].imshow(pred_mask, alpha=0.5)  # Overlay the colored mask
    ax[1].set_title('Prediction Overlay')
    ax[1].axis('off')

    # Prediction overlay
    ax[2].imshow(image, cmap='gray', clim=image_clim)  # Original image
    ax[2].imshow(binary_pred, alpha=0.5)  # Overlay the colored mask
    ax[2].set_title('Binary Prediction Overlay')
    ax[2].axis('off')

    # Ensure boxes are in the correct format (list of numpy arrays)
    if not isinstance(boxes, list):
        boxes = [boxes]

    # Draw bounding boxes on the original image
    for box in boxes:
        # Convert box tensor to numpy array if it's not already
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        
    figure_file_path = os.path.join(model_save_path, "label_QC", f"{image_name}_pred.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory

