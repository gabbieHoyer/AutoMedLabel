# Functions to get/load and display images

import numpy as np

import os
import imageio

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap

# -------------------- Color Tools --------------------
class ColorInfo():
    """
    TODO
    """
    # ---- FUNCTIONS NOT DEPENDENT ON DATA EXTENSION -----

    def set_unique_labels(self, mask):
        return [int(label) for label in np.unique(mask[mask != 0])]
    
    def set_unique_bbox_labels(self, volume_bbox_list):
        """
        - volume_bbox_list: [(slice, slice_bbox_list),...], where slice_bbox_list=[(label_id, bbox), ...]
        """
        if not isinstance(volume_bbox_list[0], tuple) and not isinstance(volume_bbox_list[0], list):
            volume_bbox_list = (0, volume_bbox_list)
        
        # Process each slice
        label_id_list = []
        for slice_idx, slice_bbox_list in volume_bbox_list:
            # Process each label and get the bounding boxes
            for label_id, bbox in slice_bbox_list:
                if label_id not in label_id_list:
                    label_id_list.append(label_id)

        return np.array(label_id_list)

    def set_image_clim(self, image):
        image_norm = 'percentile'
        # can do percentiles here
        if image_norm == 'percentile':
            return [np.percentile(image[:], 2),np.percentile(image[:], 98)]
        else:
            return [np.min(image[:]),np.max(image[:])]
    
    def set_mask_clim(self, mask):
        """
        Adjust 'clim' based on 'unique_labels' before calling 'adjust_cmap_for_labels'
        """
        unique_labels = self.set_unique_labels(mask)
        
        if not unique_labels:
            clim = [0,1]
        elif unique_labels == [0]:
            clim = [0,1]
        elif len(unique_labels) == 1:
            clim = [0, np.max(unique_labels)]
        elif len(unique_labels) > 1:
            clim = [min(unique_labels), max(unique_labels)]
        # didnt work for dhal or bacpac sag1
        return clim
    
    def set_bbox_colors(self, volume_bbox_list, cmap:str='rainbow'):
        """
        Generate colors from the colormap
        - volume_bbox_list: [(slice, slice_bbox_list),...], where slice_bbox_list=[(label_id, bbox), ...]
        """
        unique_labels = self.set_unique_bbox_labels(volume_bbox_list)
        
        # Get a color map
        colormap = cm.get_cmap(cmap)
        # Generate colors from the colormap
        label_colors = [colormap(i / len(unique_labels)) for i in range(len(unique_labels))]
    
        return dict(zip(unique_labels, label_colors))

    def adjust_cmap_for_labels(self, cmap, labels, clim):
        """
        Adjust a given colormap to fit the specified labels within the color limit (clim) range,
        ensuring each label has a consistent color across the volume.
        """
        base_cmap = plt.get_cmap(cmap)
        # If clim is provided and valid, use it to normalize
        if clim and clim[0] < clim[1]:
            norm = Normalize(vmin=clim[0], vmax=clim[1])
        else:
            # If only one label or clim not valid, adjust the range to ensure a valid normalization
            label_value = min(labels) if labels else 0  # Use 0 if labels are empty
            norm = Normalize(vmin=label_value - 0.1, vmax=label_value + 0.1)

        # Generate a new colormap from the base colormap using the normalized label values
        colors = base_cmap(np.linspace(0, 1, len(labels)))
        new_cmap = ListedColormap(colors)

        return new_cmap, norm

    def add_colorbar(self, fig, ax, labels_dict, cmap, norm, unique_labels):
        # Assuming background is already excluded from unique_labels and labels_dict

        # Calculate the boundaries and ticks using the unique labels
        boundaries = np.linspace(norm.vmin, norm.vmax, len(unique_labels) + 1)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2  # get the mid-point of boundaries

        # Create the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, boundaries=boundaries, ticks=ticks)

        # Set the tick labels
        cbar.set_ticklabels([labels_dict[label] for label in unique_labels])

        cbar.ax.tick_params(labelsize=12)
        plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')


# -------------------- Plot Volumes (3D) --------------------
        
def plot_volume(vol, save_path:str=None, cols:int=6, scale:int=3, \
        title:str=None, image_clim=[], cbar_label:str=''): 
    """
    Plot volume slices with segmentation overlay and optionally save to a file.
    Parameters:
    - vol: The volume to be displayed.
    - cols: Number of columns in the subplot grid.
    - scale: Scaling factor for each subplot.
    - title: Title for the plot.
    - save_path: Path to save the figure.
    """
    if not image_clim:
        image_clim = [np.percentile(vol[:], 2),np.percentile(vol[:], 98)]
        
    num_images = len(vol)
    rows = (num_images - 1) // cols + 1
    
    fig = plt.figure(figsize=(cols * scale, rows * scale))

    for idx in range(num_images):
        plt.subplot(rows, cols, idx+1)
        im = plt.imshow(np.squeeze(vol[idx]), cmap='gray', clim=image_clim, aspect='equal')
        plt.axis('off')

    if cbar_label:
        fig.subplots_adjust(right=0.82)
        cbar_ax = fig.add_axes([0.85, 0.128, 0.01, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_label,rotation = 270,labelpad = 20)

    if title:
        plt.suptitle(title, fontsize=20)
    
    # Save
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return
    else:
        return fig


# -------------------- Plot Segmentation Overlays --------------------
    
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
    """
    TODO
    """
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
    #print(f"GIF compiled and saved to {save_path}")

    #Optionally Cleanup temporary PNGs images and temporary directory
    for file_path in file_paths:
        os.remove(file_path)
    os.rmdir(temp_dir)
    #print("Individual slice PNGs removed after compiling GIF.")
    return