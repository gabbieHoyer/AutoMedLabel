import argparse
import os
import numpy as np
import random 
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.misc import remap_labels
from src.utils.file_management.config_handler import load_dataset_config
from src.utils.file_management.file_handler import load_standardized_npy_data 
from src.utils.file_management.path_info import pair_volume_id_paths
from src.utils.visualizers import plot_segmentation_overlay, create_segmentation_overlay_gif

class NpyFigureInfo():
    """
    TODO
    """
    # ---- FUNCTIONS SPECIFIC TO DATA EXTENSION -----
    def __init__(self, slice_selection):
        self.desired_slices = 24
        self.slice_selection = slice_selection
        return

    def load_image(self, fig_prefix_path):
        fig_path, volume_id = os.path.split(fig_prefix_path)
        data = load_standardized_npy_data(fig_path, volume_id)
        # Slice x H x W x C, remove 3 channels
        #data = data[:,:,:,0]
        return data
    
    def load_mask(self, fig_prefix_path):
        fig_path, volume_id = os.path.split(fig_prefix_path)
        data = load_standardized_npy_data(fig_path, volume_id)
        # Convert labels into int
        if len(np.unique(data)) != len(np.unique(data.astype(int))):
            raise TypeError("Non-integer values in the mask. Plotting code only allows for integer values.")
        return data
    
    def select_slices_for_display(self, data):
        if self.slice_selection == 'with_segmentation':
            # find non-zero slices
            z_index, _, _ = np.where(data > 0)
            possible_inds = np.unique(z_index)
        else:
            possible_inds = np.arange(0, np.shape(data)[0])

        num_slices = len(possible_inds)
        step_ = max(1, int(np.ceil(num_slices / self.desired_slices)))
        start = (num_slices % self.desired_slices) // 2 if num_slices > self.desired_slices else 0
        return possible_inds[range(start, num_slices, step_)]
    
    def select_files(self, image_dir:str, mask_dir:str, num_files:int=1):
        #Note: expect all files in these directories should have '.npy' extension, code does not check extension
        paired_files = pair_volume_id_paths(image_dir, mask_dir) 
        selected_pairs = random.sample(paired_files, min(len(paired_files), num_files))
        return selected_pairs
    
    def extract_save_path_info(self, fig_path:str, image_path:str):
        
        # Check for invalid save directory
        if fig_path.endswith(('.png', '.gif')):
            save_dir = os.path.dirname(fig_path)
            fname_no_ext = os.path.basename(fig_path).split(".")[0]
        else:
            save_dir = fig_path
            fname_no_ext = os.path.basename(image_path)

        return save_dir, fname_no_ext


def visualize_npz(image_dir:str, mask_dir:str, output_type:str, num_figs:int, slice_selection:str='any', output_dir:str=None, overwrite_flag:bool=False, labels_dict:dict={}, clim=None):
    """
    Create and save figures of image volumes with segmentation overlays as either an image with subplots of slices, 
    or a gif that cycles through slices.
    
    Parameters:
    - output_path: Path to the directory where figures should be saved. The base filename should reflect the mask type.
    - image_volume: The NIfTI image volume.
    - mask_volume: The segmentation mask volume (either ground truth or predicted).
    - clim: Color limits for the segmentation overlay.
    - cmap: Colormap for the segmentation overlay ('rainbow' by default).
    - labels_dict: Specifies semantic labels (values) associated with segmentation array (keys). 
        Ex. {0: 'background', 1: 'anatomy'} 
    - output_type: Specifies which figures to generate. Options include: '2D_overlay', 'gif', or ['2D_overlay', 'gif']
    - force_flag: True or False. Whether to save figure if file already exists.
    """

    FigDataFns = NpyFigureInfo(slice_selection)

    selected_pairs = FigDataFns.select_files(image_dir, mask_dir, num_figs)
    print(output_dir)
    for image_prefix_path, mask_prefix_path in tqdm(selected_pairs):

        image = FigDataFns.load_image(image_prefix_path)
        mask = FigDataFns.load_mask(mask_prefix_path)
        slices = FigDataFns.select_slices_for_display(mask)
        
        save_dir, fname_no_ext = FigDataFns.extract_save_path_info(output_dir, image_prefix_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if '2D_overlay' in output_type:
            # Set up the path for saving the overlay plot
            plot_savepath = os.path.join(save_dir, f"{fname_no_ext}_overlay.png")
            
            # Check whether to overwrite existing data
            if not os.path.isfile(plot_savepath) or overwrite_flag == True:
                # Generate and save the overlay plot with label dictionary support
                plot_segmentation_overlay(vol= image[slices,:,:], 
                                        seg= mask[slices,:,:], 
                                        save_path=plot_savepath, 
                                        seg_clim=clim, 
                                        cmap='rainbow', 
                                        title=fname_no_ext, 
                                        labels_dict=labels_dict,
                                        ) 

        if 'gif' in output_type:
            # Set up the directory and filename for the GIF
            gif_filepath = os.path.join(save_dir, f"{fname_no_ext}.gif")

            # Check whether to overwrite existing data
            if not os.path.isfile(gif_filepath) or overwrite_flag == True:
                
                # Create and save the GIF
                create_segmentation_overlay_gif(image_volume=image[slices,:,:], 
                        segmentation=mask[slices,:,:], 
                        save_path=gif_filepath, 
                        cmap='rainbow', 
                        )

def npy_visualization(config_name):
    """
    TODO
    """
    cfg = load_dataset_config(config_name, root)

    #Modify target labels
    filtered_labels_dict = remap_labels(cfg.get('mask_labels', {}),
                                        cfg.get('target_label_id', None)  # Could be None, an int, or a list
                                        )

    #Call visualization script
    visualize_npz(
                    os.path.join(cfg.get('npy_dir'), "imgs"), 
                    os.path.join(cfg.get('npy_dir'), "gts"), 
                    cfg.get("npy_fig_cfg", "").get('fig_type'),  # Default to '2D_overlay' if not specified
                    cfg.get("npy_fig_cfg", "").get('num_figs'),
                    cfg.get("nifti_fig_cfg", "").get('slice_selection', 'any'),
                    cfg.get("npy_fig_cfg", "").get('fig_path'), 
                    cfg.get("overwrite_existing_flag"),
                    filtered_labels_dict,  # Assuming cfg is your loaded configuration dict 
                    cfg.get("npy_fig_cfg", "").get('clim', None),
                    )
    pass


if __name__ == "__main__":
    """
    Conditionally generate and save overlay plots and/or create a high-resolution GIF from an image volume and mask volume,
    based on the specified output type. This is done twice: once for ground truth masks, and once for model-predicted masks.
    """
    parser = argparse.ArgumentParser(description="Visualize NPZ volume inference cases.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()


    config_name = args.config_name + '.yaml'  # Get the config file name from the command line argument

    npy_visualization(config_name)
