
import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

from .color_utils import ColorInfo

# -------------------- Plot Segmentation Overlays --------------------

def upscale_volume_mask_cv2(mask: np.ndarray, target_hw: tuple) -> np.ndarray:
    """
    Upscale a 3D mask volume (or a single 2D mask) to the target height and width
    without converting the entire volume to a torch tensor.
    
    Args:
        mask (np.ndarray): A 2D array (H, W) or a 3D array (num_slices, H, W).
        target_hw (tuple): Desired output size as (target_height, target_width).
        
    Returns:
        np.ndarray: The upscaled mask (same number of slices if 3D, or a single 2D array).
    """
    if mask.ndim == 2:
        # For a single 2D mask
        upscaled = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
        return upscaled.astype(mask.dtype)
    elif mask.ndim == 3:
        # For a volume of 2D masks
        upscaled_slices = [
            cv2.resize(mask[i, :, :], (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR).astype(mask.dtype)
            for i in range(mask.shape[0])
        ]
        return np.stack(upscaled_slices, axis=0)
    else:
        raise ValueError("Mask must be a 2D or 3D numpy array.")


def plot_segmentation_overlay(vol, seg, save_path:str=None, cols:int=6, scale:int=3, \
        seg_clim=None, cmap='rainbow', title:str=None, labels_dict={}): 
    """
    Plot volume slices with segmentation overlay and optionally save to a file.
    Parameters:
    - vol: The volume to be displayed.
    - seg: The segmentation mask to overlay.
    - cols: Number of columns in the subplot grid.
    - scale: Scaling factor for each subplot.
    - clim: Color limits for the segmentation overlay.
    - cmap: Colormap for the segmentation overlay ('rainbow' by default).
    - title: Title for the plot.
    - save_path: Path to save the figure.
    - labels_dict: Specifies semantic labels (values) associated with segmentation array (keys). 
        Ex. {0: background, 1: anatomy} 
    """
    # Get the target dimensions from the image volume
    target_hw = (vol.shape[1], vol.shape[2])
    
    # If the segmentation mask dimensions don't match, upscale them.
    # Here, assuming seg is a 3D volume of shape (num_slices, H, W)
    if seg.shape[-2:] != target_hw:
        seg = upscale_volume_mask_cv2(seg, target_hw)

    #Define properties for plot colors
    ColorFns = ColorInfo()
    
    image_clim = ColorFns.set_image_clim(vol)
    #if mask_clim is None:
    seg_clim = ColorFns.set_mask_clim(seg)
    unique_labels = ColorFns.set_unique_labels(seg)
    adjusted_cmap, norm = ColorFns.adjust_cmap_for_labels(cmap, unique_labels, seg_clim)
    
    num_images = len(vol)
    rows = (num_images - 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
    axes = axes.flatten()
    
    for idx in range(num_images):
        ax = axes[idx]
        ax.imshow(np.squeeze(vol[idx]), cmap='gray', clim=image_clim, aspect='equal')
        overlay = np.ma.masked_where(seg[idx] == 0, seg[idx])
        im = ax.imshow(overlay, cmap=adjusted_cmap, norm=norm, alpha=0.4, clim=seg_clim)
        ax.axis('off')
    
    # Adjust for empty subplots if num_images is not a multiple of cols
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=20)
    
    if (labels_dict) and (unique_labels != []) and (num_images > 0):
        ColorFns.add_colorbar(fig, axes, labels_dict, adjusted_cmap, norm, unique_labels)
    
    # Save
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return
    else:
        return fig

def create_segmentation_overlay_gif(image_volume, segmentation, save_path, cmap='rainbow'):
    """
    Create a GIF from a series of images with segmentation overlay.
    """
    # Get the target dimensions from the image volume
    target_hw = (image_volume.shape[1], image_volume.shape[2])
    
    # If the segmentation mask dimensions don't match, upscale them.
    # Here, assuming segmentation is a 3D volume of shape (num_slices, H, W)
    if segmentation.shape[-2:] != target_hw:
        segmentation = upscale_volume_mask_cv2(segmentation, target_hw)

    #Define properties for plot colors
    ColorFns = ColorInfo()
    image_clim = ColorFns.set_image_clim(image_volume)
    #if mask_clim is None:
    seg_clim = ColorFns.set_mask_clim(segmentation)
    unique_labels = ColorFns.set_unique_labels(segmentation)
    adjusted_cmap, norm = ColorFns.adjust_cmap_for_labels(cmap, unique_labels, seg_clim)

    # Create a temporary directory for files that will be combined into .gif
    temp_dir = save_path.replace('.gif','') + '_temp_images'
    os.makedirs(temp_dir, exist_ok=True)
    # Save images to temporary directory 
    num_images = image_volume.shape[0]
    file_paths = []
    for idx in range(num_images):
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(image_volume[idx, :, :]), cmap='gray', clim=image_clim)
        overlay = np.ma.masked_where(segmentation[idx, :, :] == 0, segmentation[idx, :, :])
        ax.imshow(overlay, cmap=adjusted_cmap, norm=norm, alpha=0.4, clim=seg_clim)
        ax.axis('off')
        
        temp_file_path = os.path.join(temp_dir, f"frame_{idx:03}.png")
        plt.savefig(temp_file_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        file_paths.append(temp_file_path)

    #Create the gif using images in temporary directory
    with imageio.get_writer(save_path, mode='I') as writer:
        for file_path in file_paths:
            image = imageio.imread(file_path)
            writer.append_data(image)

    #Cleanup temporary images and temporary directory
    for file_path in file_paths:
        os.remove(file_path)
    os.rmdir(temp_dir)
    return

# -------------------- Plot Bbox Overlays --------------------

def show_box(box, ax, edgecolor='blue'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))

def plot_bbox_overlay(vol, volume_bbox_list, save_path:str=None, cols:int=6, scale:int=3, \
        cmap='rainbow', title:str=None, labels_dict=None): 
    """
    Plot volume slices with segmentation overlay and optionally save to a file.
    Parameters:
    - vol: The volume to be displayed.
    - seg: The segmentation mask to overlay.
    - cols: Number of columns in the subplot grid.
    - scale: Scaling factor for each subplot.
    - clim: Color limits for the segmentation overlay.
    - cmap: Colormap for the segmentation overlay ('rainbow' by default).
    - title: Title for the plot.
    - save_path: Path to save the figure.
    - labels_dict: Specifies semantic labels (values) associated with segmentation array (keys). 
        Ex. {0: background, 1: anatomy} 
    """

    #Define properties for plot colors
    ColorFns = ColorInfo()
    image_clim = ColorFns.set_image_clim(vol)
    label_color_dict = ColorFns.set_bbox_colors(volume_bbox_list, cmap)
    unique_labels = ColorFns.set_unique_bbox_labels(volume_bbox_list)
    
    num_images = len(vol)
    rows = (num_images - 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
    axes = axes.flatten()
    volume_bbox_dict = dict(volume_bbox_list)
    for slice_idx in range(num_images):
        ax = axes[slice_idx]
        ax.imshow(np.squeeze(vol[slice_idx]), cmap='gray', clim=image_clim) #, aspect='equal'
        
        # Draw each bounding box with a color from the colormap
        if slice_idx in volume_bbox_dict.keys(): #keys are the slice indexes
            slice_bbox_list = volume_bbox_dict[slice_idx]
            
            for label_id, bbox in slice_bbox_list:
                show_box(bbox, ax, edgecolor=label_color_dict[label_id])
        ax.axis('off')

    # Adjust for empty subplots if num_images is not a multiple of cols
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    if title:
        plt.suptitle(title, fontsize=20)
    
    # Save
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return
    else:
        return fig

def create_bbox_overlay_gif(image_volume, volume_bbox_list, save_path, cmap='rainbow'):

    # Create a temporary directory for files that will be combined into .gif
    temp_dir = save_path.replace('.gif','') + '_temp_images'
    os.makedirs(temp_dir, exist_ok=True)

    #Define properties for plot colors
    ColorFns = ColorInfo()
    image_clim = ColorFns.set_image_clim(image_volume)
    label_color_dict = ColorFns.set_bbox_colors(volume_bbox_list)

    # Save images to temporary directory 
    num_images = image_volume.shape[0]
    volume_bbox_dict = dict(volume_bbox_list)
    file_paths = []
    for slice_idx in range(num_images):
        image = np.squeeze(image_volume[slice_idx, :, :])

        # Plot image with overlaid bounding box
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', clim=image_clim) 
        # Draw each bounding box with a color from the colormap
        if slice_idx in volume_bbox_dict.keys(): #keys are the slice indexes
            slice_bbox_list = volume_bbox_dict[slice_idx]
            for label_id, bbox in slice_bbox_list:
                show_box(bbox, ax, edgecolor=label_color_dict[label_id])
        ax.axis('off')
        
        temp_file_path = os.path.join(temp_dir, f"frame_{slice_idx:03}.png")
        plt.savefig(temp_file_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        file_paths.append(temp_file_path)

    #Create the gif using images in temporary directory
    # Faster playback for more slices, otherwise slower playback for fewer slices
    num_slices = len(file_paths)
    fps = 10 if num_slices > 20 else 2
    #Compile images
    with imageio.get_writer(save_path, mode='I', fps=fps, loop=0) as writer:
        for file_path in file_paths:
            image = imageio.imread(file_path)
            writer.append_data(image)

    #Optionally Cleanup temporary PNGs images and temporary directory
    for file_path in file_paths:
        os.remove(file_path)
    os.rmdir(temp_dir)
    #print("Individual slice PNGs removed after compiling GIF.")
    return