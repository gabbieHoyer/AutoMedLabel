
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from skimage import measure

import torch
import torch.nn.functional as F

# ------------------- VISUALIZATION TOOLS ------------------- #

def set_image_clim(image):
    image_norm = 'percentile'
    # can do percentiles here
    if image_norm == 'percentile':
        return [np.percentile(image[:], 2),np.percentile(image[:], 98)]
    else:
        return [np.min(image[:]),np.max(image[:])]
    
def set_image_norm(image):
    low = np.percentile(image, 2)
    high = np.percentile(image, 98)
    return Normalize(vmin=low, vmax=high)

def compute_image_clim_log(image, p_low=2, p_high=98):
    log_img = np.log1p(np.clip(image, a_min=0, a_max=None))  # log(1 + x) to avoid log(0)
    low, high = np.percentile(log_img, [p_low, p_high])
    return Normalize(vmin=low, vmax=high)

def compute_clamped_percentiles(image, p_low=2, p_high=98, min_range=50, max_range=2000):
    flat = image.flatten()
    low, high = np.percentile(flat, [p_low, p_high])
    actual_range = high - low
    if actual_range < min_range:
        mid = (high + low) / 2
        low = mid - min_range/2
        high = mid + min_range/2
    elif actual_range > max_range:
        mid = (high + low) / 2
        low = mid - max_range/2
        high = mid + max_range/2
    return Normalize(vmin=low, vmax=high)
    
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

def map_labels_to_contours_qc(pred_mask, mask_labels):
    """
    Generate a dictionary mapping each label to its corresponding contours and colors.
    """
    contours_dict = {}

    # Define the colormap to get consistent colors for each label
    color_map = plt.get_cmap('rainbow', len(mask_labels) - 1)  # Exclude background

    for label_value, label_name in mask_labels.items():
        if label_value == 0:  # Skip the background
            continue
        
        # Find contours for the specific class label
        mask = (pred_mask == label_value)
        contours = measure.find_contours(mask, 0.5)  # Detect the contours in the binary mask
        
        # Get the consistent color for this label
        color = color_map((label_value - 1) / (len(mask_labels) - 1))  # Normalize label_value for colormap
        
        # Store contours and corresponding color
        contours_dict[label_value] = {
            "contours": contours,
            "color": color  # Store the color for each label
        }

    return contours_dict

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
    
def add_colorbar_h(fig, ax, mask_labels):
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

    # Dynamically adjust the aspect ratio based on number of labels
    if len(unique_labels) <= 10:
        aspect = 50  # Stretch bar when few labels
    else:
        aspect = 20 + len(unique_labels)  # Adjust aspect for more labels

    # Add colorbar to the figure, dynamically adjusting aspect and fraction
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation='horizontal', aspect=aspect, pad=0.1, fraction=0.05)

    # Set the ticks and labels based on the unique mask labels
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([mask_labels[label] for label in unique_labels])

def upscale_mask(mask, target_hw):
    """
    Upscale a mask tensor to the target height and width.

    Args:
        mask (torch.Tensor): A mask tensor assumed to be 256×256.
        target_hw (tuple): Desired output size as (height, width), e.g. (1024, 1024).

    Returns:
        torch.Tensor: The upscaled mask as a 2D tensor.
    """
    # Convert to a Torch tensor if it's not already
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    
    # Ensure the mask has shape (N, C, H, W)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    
    upscaled = F.interpolate(mask.float(), size=target_hw, mode='bilinear', align_corners=False)
    # Remove batch and channel dimensions
    upscaled = upscaled.squeeze(0).squeeze(0)
    return upscaled 



