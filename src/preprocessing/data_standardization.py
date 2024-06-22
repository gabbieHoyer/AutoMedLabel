# Dedicated to converting various file formats to NIfTI.
# Would use libraries like pydicom and nibabel.
# This is where you'd place functions like convert_to_nifti

import os
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    dotenv=True, # load environment variables from .env if exists in root directory
)

from src.utils.file_management.config_handler import load_dataset_config
from src.utils.file_management.file_handler import save_nifti
from src.preprocessing.data_loaders.dataset_handler import get_data, get_combined_data, validate_data
from src.utils.file_management.path_info import extract_data_paths

import numpy as np
# Define the main conversion function for MRI images or masks
def convert_to_nifti(data_dir, nifti_output_dir:str, columns_to_process=None, \
                     key:str='', data_transforms:dict={}, expected_properties:dict={}, \
                     overwrite_flag:bool=True, no_dicom_ext_flag:bool=False, \
                     subject_test_sample:int=None):
    """
    Convert data to NIfTI format.

    Args:
        nifti_dir: Output directory for NIfTI files.
        columns: List of columns specifying file paths to convert.
        data_type: Flag indicating the type of data ('mri' or 'mask').
    """

    # Make output directory if it does not exist
    if not os.path.exists(nifti_output_dir):
        os.makedirs(nifti_output_dir)
    
    data_paths_info = extract_data_paths(data_dir, columns_to_process)
    #data_paths_info = data_paths_info[35:36] #38 mask problem
    # Apply the slicing only if subject_test_sample is provided and is a positive integer
    if subject_test_sample and isinstance(subject_test_sample, int) and subject_test_sample > 0:
        data_paths_info = data_paths_info[:subject_test_sample]
    
    # Iterate though each subject to load and save data
    for data_paths, subject_id in tqdm(data_paths_info):
        
        save_path = os.path.join(nifti_output_dir, f"{subject_id}.nii.gz")
        # Check whether to overwrite existing data
        if (overwrite_flag == False) and os.path.isfile(save_path):
            continue
        
        # Load data from a single location or combine data from multiple locations, & save
        if isinstance(data_paths, str):
            data = get_data(data_paths, key, data_transforms, no_dicom_ext_flag)
        elif len(data_paths)==1:
            data = get_data(data_paths[0], key, data_transforms, no_dicom_ext_flag)
        elif len(data_paths)>1:
            data = get_combined_data(data_paths, key, data_transforms)
        else:
            raise ValueError('Invalid data path.')
        
        if type(expected_properties) == dict:
            #TODO
            validate_data(data, expected_properties)
        save_nifti(data, save_path)
    
    return 

    
def data_standardization(config_name):
    
    cfg = load_dataset_config(config_name, root)

    def check_path_exists(cfg, dir_key):
        return cfg.get(dir_key) and os.path.exists(cfg[dir_key])
    #import pdb; pdb.set_trace()
    if check_path_exists(cfg, "image_data_paths"):
        convert_to_nifti(cfg.get("image_data_paths"),
                         cfg.get("nifti_image_dir"),
                         cfg.get("image_column"),
                         cfg.get("image_config", "").get("key", ""),
                         cfg.get("image_config", "").get("transforms", ""),
                         cfg.get("image_config", "").get("data_properties", ""),
                         cfg.get("overwrite_existing_flag"),
                         cfg.get("no_dicom_extension_flag", False),
                         cfg.get("subject_test_sample", ""),
                         )
    #import pdb; pdb.set_trace()
    if check_path_exists(cfg, "mask_data_paths"):
        convert_to_nifti(cfg.get("mask_data_paths"),
                         cfg.get("nifti_mask_dir"),
                         cfg.get("mask_columns", cfg.get("mask_column")),
                         cfg.get("mask_config", "").get("key", ""),
                         cfg.get("mask_config", "").get("transforms", ""),
                         cfg.get("mask_config", "").get("data_properties", ""),
                         cfg.get("overwrite_existing_flag"),
                         subject_test_sample = cfg.get("subject_test_sample", ""),
                         )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert medical data to NIfTI format.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()

    config_name = args.config_name + '.yaml'

    data_standardization(config_name)