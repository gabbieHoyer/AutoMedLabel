
import os
import numpy as np

from string import Template

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.file_management.path_info import volume_id_file_paths_in_dir

# Do not modify - All functions are needed because they are imported to other scripts
from src.utils.file_management.file_loaders_savers import (
    load_nifti, save_nifti,
    load_json, save_json, save_parquet,
    load_npy, load_npz,
    )

# -------------------- DATA LOADING FUNCTIONS --------------------

def load_data(file_path:str, key:str=''):
    """
    Load data. This function supports npy, npz, and NIfTI data formats.
    """
    # Use the .suffix and .suffixes properties of Path objects
    if file_path.endswith('.npz'):
        volume = load_npz(file_path, key=key) # Adjust based on actual key
    elif file_path.endswith('.npy'):
        volume = load_npy(file_path, key=key) # Adjust based on actual key
    elif file_path.endswith('.nii') or file_path.endswith('.nii.gz') :  # Handling .nii.gz files
        volume = load_nifti(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return volume

def load_standardized_npy_data(npy_files_dir:str, volume_id:str):
    """Load npy files for a subject"""

    # Get extension of files in directory
    if not os.path.isdir(npy_files_dir):
        return ValueError(f"Directory does not exist: {npy_files_dir}")
    
    slice_files = volume_id_file_paths_in_dir(npy_files_dir, volume_id)
    if not slice_files:
        return ValueError(f"File name prefix does not exist: {volume_id}")
    
    # Load each file to create volume
    slices = []
    for file_path in sorted(slice_files):
        slice_data = np.load(file_path, "r", allow_pickle=True)
        slices.append(slice_data)
    data = np.stack(slices, axis=0)
    return data 

# -------------------- DATA SAVING FUNCTIONS --------------------
    
def save_metadata(data, save_path:str):
    """ 
    Save metadata to a parquet or json file.
    """
    #TODO check if metadata exists or force flag

    save_dir = os.path.dirname(save_path)
    # Make output directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if save_path.endswith('.json'):
        return save_json(data, save_path)
    elif save_path.endswith('.parquet'):
        return save_parquet(data, save_path)
    
    return False



# def load_volume_and_spacing(file_path):
#     # Ensure file_path is a Path object
#     file_path = Path(file_path)

#     # Use the .suffix and .suffixes properties of Path objects
#     if file_path.suffix == '.npz':
#         data = np.load(str(file_path), allow_pickle=True)
#         volume = data['x']  # Adjust based on actual key
#         spacing = data.get('spacing', None)  # Adjust as needed
#     elif file_path.suffixes == ['.nii', '.gz']:  # Handling .nii.gz files
#         nii_img = nib.load(str(file_path))
#         volume = nii_img.get_fdata()
#         spacing = nii_img.header.get_zooms()
#     else:
#         raise ValueError(f"Unsupported file format: {file_path}")

#     return volume, spacing