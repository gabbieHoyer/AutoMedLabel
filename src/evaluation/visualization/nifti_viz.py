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

from src.utils.file_management.config_handler import load_dataset_config
from src.utils.file_management.file_loaders_savers import load_nifti
from src.utils.file_management.path_info import pair_files, file_without_extension_from_path 

from src.utils.visualizers import plot_segmentation_overlay, create_segmentation_overlay_gif

class NiftiFigureInfo():
    """
    TODO
    """
    def __init__(self, slice_selection):
        self.desired_slices = 24
        self.slice_selection = slice_selection
        return

    # ---- FUNCTIONS SPECIFIC TO DATA EXTENSION -----
    def make_values_valid(self, vol):
        """
        Replace invalid values in the volume and segmentation mask
        """
        valid_vol = np.copy(vol)
        #Check for non-finite values in the volume and segmentation mask
        if not np.all(np.isfinite(valid_vol)):
            # Replace NaNs and infs with zero or a specified value
            valid_vol =  np.nan_to_num(vol) 
        # If seg is a MaskedArray, fill the masked values with a fill value (e.g., 0)
        if np.ma.is_masked(valid_vol):
            # Replace masked values with zero
            valid_vol = valid_vol.filled(0) 
        return valid_vol
    
    def load_image(self, fig_path):
        data = load_nifti(fig_path)
        data = self.make_values_valid(data)
        return data
    
    def load_mask(self, fig_path):
        data = load_nifti(fig_path)
        
        # Convert labels into int
        if len(np.unique(data)) != len(np.unique(data.astype(int))):
            raise TypeError("Non-integer values in the mask. Plotting code only allows for integer values.")
        data = self.make_values_valid(data.astype(int))
        
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
        
    
    # def select_files(self, image_dir:str, mask_dir:str, num_files:int=1):
    #     paired_files = pair_files(image_dir, mask_dir, ('.nii', '.nii.gz'))
    #     selected_pairs = random.sample(paired_files, min(len(paired_files), num_files))
    #     selected_pairs = paired_files #TODO uncomment to process all files
    #     return selected_pairs

    def select_files(self, image_dir: str, mask_dir: str, num_files: int = 1):
        paired_files = pair_files(image_dir, mask_dir, ('.nii', '.nii.gz'))
        
        if num_files == "full":
            selected_pairs = paired_files
        else:
            selected_pairs = random.sample(paired_files, min(len(paired_files), num_files))
            
        return selected_pairs
    
    def extract_save_path_info(self, fig_path:str, image_path:str):
        
        # Check for invalid save directory
        if fig_path.endswith(('.png', '.gif')):
            save_dir = os.path.dirname(fig_path)
            fname_no_ext = os.path.basename(fig_path).split(".")[0]
        else:
            save_dir = fig_path
            fname_no_ext = file_without_extension_from_path(os.path.basename(image_path))

        return save_dir, fname_no_ext

def visualize_nifti(image_dir:str, mask_dir:str, output_type:str, num_figs:int, slice_selection:str='any', output_dir:str=None, overwrite_flag:bool=False, labels_dict:dict={}, clim=None):
    """
    Create and save figures of image volumes with segmentation overlays as either a image with subplots of slices, 
    or a gif that cycles through slices.
    
    Parameters:
    - output_path: Full path to save figure.
    - clim: Color limits for the segmentation overlay.
    - cmap: Colormap for the segmentation overlay ('rainbow' by default).
    - labels_dict: Specifies semantic labels (values) associated with segmentation array (keys). 
        Ex. {0: background, 1: anatomy} 
    - output_type: Specifies which figures to generate. Options include: '2D_overlay', 'gif', or ['2D_overlay', 'gif']
    - overwrite_flag: True or False. Whether to save figure if file already exists.
    """

    FigDataFns = NiftiFigureInfo(slice_selection)

    selected_pairs = FigDataFns.select_files(image_dir, mask_dir, num_figs)
    
    for image_path, mask_path in tqdm(selected_pairs):    
        image = FigDataFns.load_image(image_path)
        mask = FigDataFns.load_mask(mask_path)
        slices = FigDataFns.select_slices_for_display(mask)
        
        save_dir, fname_no_ext = FigDataFns.extract_save_path_info(output_dir, image_path)
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

        #print(labels_dict)
    return


def nifti_visualization(config_name):
    
    cfg = load_dataset_config(config_name, root)

    visualize_nifti(
                    cfg.get('nifti_image_dir'), 
                    cfg.get('nifti_mask_dir'), 
                    cfg.get("nifti_fig_cfg", "").get('fig_type'),  # Default to '2D_overlay' if not specified
                    cfg.get("nifti_fig_cfg", "").get('num_figs'),
                    cfg.get("nifti_fig_cfg", "").get('slice_selection', 'any'),
                    cfg.get("nifti_fig_cfg", "").get('fig_path'), 
                    cfg.get("overwrite_existing_flag"),
                    cfg.get('mask_labels', None),  # Assuming cfg is your loaded configuration dict 
                    cfg.get("nifti_fig_cfg", "").get('clim', None),
                    )
    return

if __name__ == "__main__":
    """
    Conditionally generate and save overlay plots and/or create a high-resolution GIF from an image volume and mask volume,
    based on the specified output type.
    """
    parser = argparse.ArgumentParser(description="Convert medical data to NIfTI format.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()
    
    config_name = args.config_name + '.yaml'

    nifti_visualization(config_name)



