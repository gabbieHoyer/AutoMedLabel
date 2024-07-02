import argparse
import os
from tqdm import tqdm
import pandas as pd
import pydicom

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    dotenv=True, # load environment variables from .env if exists in root directory
)

from src.utils.file_management.config_handler import load_dataset_config
from src.utils.file_management.file_handler import load_json, save_metadata
from src.utils.file_management.path_info import extract_data_paths, volume_ids_in_dir, volume_id_file_paths_in_dir
from src.preprocessing.metadata_management.metadata_extractors import generate_subject_metadata, extract_header_info, generate_slice_info_for_subject, summarize_unique_dicom_data
from src.preprocessing.metadata_management.metadata_split import preprocess_subjects, stratify_and_sample


def create_metadata_for_statistics(mri_dicom_dir:str, dicom_values:dict, dataset_info:dict, \
                                   columns_to_process=None, force_dicom:bool=False):
    """
    Function to create primary metadata for stats (Operation A)

    Params:
    - mri_dicom_dir: Can be a directory or a CSV file
    - dicom_values: Dictionary of standardized and actual keys to metadata.
    - dataset_info: Additional characteristics from dataset.

    - columns_to_process: List of columns specifying file path to extract metadata.
    - force_dicom: Whether to force the processing of all files in the mri_dicom_dir as DICOM files, regardless of their file extension

    Returns:
    - dataset_metadata: Dictionary containing volume-specific dataset info and demographics, and
            slice-specific dicom header metadata for all subjects in the dataset.
    """

    def find_dicom_file(file_path, force_dicom:bool=False):
        """
        If file is a dicom, returns filename without extension. 
        """
        allowed_extensions = ['.dcm', '.dcm.gz', '.DCM', '.DCM.GZ']  # Extensions to check for if not forcing DICOM processing

        # Do not consider hidden files
        if file_path.startswith('.'):
            slice_name = None
        # Assume the file is a DICOM file if force_dicom is True, regardless of the extension
        elif force_dicom:
            # Use the entire filename as slice_name when extensions are not considered
            slice_name = f"{file_path}"
        elif any([file_path.endswith(ext) for ext in allowed_extensions]):
            # Trim the known DICOM extensions to get the slice_name
            for ext in allowed_extensions:
                if file_path.endswith(ext):
                    remove_ext = ext
            slice_name = file_path.rsplit(remove_ext, 1)[0]
        else:
            slice_name = None

        return slice_name
    
    # Extract paths to dicom directories and subject ids
    data_paths_info = extract_data_paths(mri_dicom_dir, columns_to_process)
    # data_paths_info = data_paths_info[0:5]  ##### this is a check and must be removed
    
    # Load and save data
    dataset_metadata = {}
    # Iterate through each subject/scan dicom folder for each image volume
    for subject_dicom_folders, subject_id in tqdm(data_paths_info):
        subject_dicom_folder = subject_dicom_folders[0]
        #Only process directories
        if not os.path.isdir(subject_dicom_folder):
            continue

        # Iterate through each dicom file in folder (slice images)
        found_files = False
        for entry in os.scandir(subject_dicom_folder):
            file_path = entry.path

            # Only process files, skip directories and misc.
            if not os.path.isfile(file_path):
                continue
            # Skip processing this file as it doesn't have a known DICOM extension and force_dicom is False
            slice_name = find_dicom_file(os.path.basename(file_path), force_dicom)
            if not slice_name:
                continue
            found_files = True
            
            # Extract metadata
            subject_csv_config = dataset_info['csv_config'].copy()
            if subject_csv_config:
                subject_csv_config["subject_id"] = subject_id
            
            slice_metadata, demographics = extract_header_info(file_path, dicom_values, subject_csv_config)
            
            # Add subject & slice metadata to dict for all subjects, scans, and slices
            if subject_id not in dataset_metadata:
                dataset_metadata[subject_id] = {
                    'Dataset': dataset_info['dataset_name'],
                    'Anatomy': dataset_info['anatomy'],
                    'slices': {},
                    'Demographics': demographics  # Include demographics
                }

            dataset_metadata[subject_id]['slices'][slice_name] = slice_metadata

        if not found_files:
            print(f"Warning: No DICOM files found for subject '{subject_id}' in the path '{subject_dicom_folder}'.")

    return dataset_metadata


def save_dicom_metadata_summary(complete_dicom_metadata, file_name):

    # open a file in write mode
    with open(file_name, 'w') as file:
        # write variables using repr() function
        for key, value in complete_dicom_metadata.items():
            file.write(key+" = "+repr(value)+'\n\n')
    return True

def create_metadata_for_acquisition_info(mri_dicom_dir:str, dicom_values:dict, dataset_info:dict, \
                                   columns_to_process=None, force_dicom:bool=False):
    """
    Function to create primary metadata for stats (Operation A)

    Params:
    - mri_dicom_dir: Can be a directory or a CSV file
    - dicom_values: Dictionary of standardized and actual keys to metadata.
    - dataset_info: Additional characteristics from dataset.

    - columns_to_process: List of columns specifying file path to extract metadata.
    - force_dicom: Whether to force the processing of all files in the mri_dicom_dir as DICOM files, regardless of their file extension

    Returns:
    - dataset_metadata: Dictionary containing volume-specific dataset info and demographics, and
            slice-specific dicom header metadata for all subjects in the dataset.
    """

    def find_dicom_file(file_path, force_dicom:bool=False):
        """
        If file is a dicom, returns filename without extension. 
        """
        allowed_extensions = ['.dcm', '.dcm.gz', '.DCM', '.DCM.GZ']  # Extensions to check for if not forcing DICOM processing

        # Do not consider hidden files
        if file_path.startswith('.'):
            slice_name = None
        # Assume the file is a DICOM file if force_dicom is True, regardless of the extension
        elif force_dicom:
            # Use the entire filename as slice_name when extensions are not considered
            slice_name = f"{file_path}"
        elif any([file_path.endswith(ext) for ext in allowed_extensions]):
            # Trim the known DICOM extensions to get the slice_name
            for ext in allowed_extensions:
                if file_path.endswith(ext):
                    remove_ext = ext
            slice_name = file_path.rsplit(remove_ext, 1)[0]
        else:
            slice_name = None

        return slice_name

    # Extract paths to dicom directories and subject ids
    data_paths_info = extract_data_paths(mri_dicom_dir, columns_to_process)
    # data_paths_info = data_paths_info[0:5]  ##### this is a check and must be removed
    
    # Load and save data
    dicom_metadata = {}
    demographic_metadata = {}
    complete_dicom_metadata = {}
    # Iterate through each subject/scan dicom folder for each image volume
    for subject_dicom_folders, subject_id in tqdm(data_paths_info):
        subject_dicom_folder = subject_dicom_folders[0]
        #Only process directories
        if not os.path.isdir(subject_dicom_folder):
            continue

        # Iterate through each dicom file in folder (slice images)
        found_files = False

        for entry in os.scandir(subject_dicom_folder):
            file_path = entry.path

            # Only process files, skip directories and misc.
            if not os.path.isfile(file_path):
                continue
            # Skip processing this file as it doesn't have a known DICOM extension and force_dicom is False
            slice_name = find_dicom_file(os.path.basename(file_path), force_dicom)
            
            if slice_name:
                found_files = True
                break
            
        # Extract metadata from one slice for the patient
        subject_csv_config = dataset_info['csv_config'].copy()
        if subject_csv_config:
            subject_csv_config["subject_id"] = subject_id
        
        slice_metadata, demographics = extract_header_info(file_path, dicom_values, subject_csv_config)
        
        # Add subject & slice metadata to dict for all subjects, scans, and slices
        if subject_id not in demographic_metadata:
            demographic_metadata[subject_id] = demographics  # Include demographics

        dicom_metadata[subject_id] = slice_metadata

        if not found_files:
            print(f"Warning: No DICOM files found for subject '{subject_id}' in the path '{subject_dicom_folder}'.")

    dicom_metadata = pd.DataFrame.from_dict(dicom_metadata, orient='index')

    complete_dicom_metadata['file_path'] = file_path
    complete_dicom_metadata['dicom_metadata'] = pydicom.dcmread(file_path)
    unique_dicom_metadata_summary = summarize_unique_dicom_data(dicom_metadata)
    complete_dicom_metadata = {**complete_dicom_metadata, **unique_dicom_metadata_summary}

    return dicom_metadata, demographic_metadata, complete_dicom_metadata

def create_relevant_metadata_with_splits(nifti_image_dir:str, nifti_mask_dir:str, split_ratios:dict, \
                                        dataset_info:dict, additional_metadata_file:str=None):
    """
    Function to create primary metadata relevant to ML/AI data splitting (train/val/test) (Operation B)
    The metadata includes split information and a subset of all possible metadata for every subject in dataset.
    
    Params:
    - nifti_image_dir: Folder containing NIfTI images.
    - nifti_mask_dir: Folder containing NIfTI masks.
    - split_ratios: Train, Val, Test split ratios. Ex. {'train': 0.7, 'val': 0.15, 'test': 0.15}
    - dataset_info: Additional characteristics from dataset.
    - additional_metadata_file: #TODO 

    Returns:
    - dataset_metadata: Dictionary containing relevant metadata for all subjects in the dataset,
            including balanced split information.
    """
    
    def load_additional_metadata(additional_metadata_file:str):
        if os.path.exists(additional_metadata_file):
            return load_json(additional_metadata_file)
        else:
            print(f"Warning: Additional metadata file not found, {additional_metadata_file}")
            return {}
        
    allowed_extensions = ['.nii', '.nii.gz']  # Extensions to check for if not forcing DICOM processing

    # Load additional metadata if exists
    additional_metadata = {}
    if additional_metadata_file:
        additional_metadata = load_additional_metadata(additional_metadata_file)
    
    # Iterate through subject_ids to summarize metadata
    dataset_metadata = {}
    data_paths_info = extract_data_paths(nifti_image_dir)
    for image_nifti_paths, subject_id in data_paths_info:
        image_nifti_path = image_nifti_paths[0]
        # Only consider nii files
        if not any([image_nifti_path.endswith(ext) for ext in allowed_extensions]):
            continue
        
        mask_nifti_path = image_nifti_path.replace(nifti_image_dir, nifti_mask_dir)
        # Verify files exist
        if not os.path.exists(image_nifti_path) or not os.path.exists(mask_nifti_path):
            print(f"Files for subject {subject_id} not found.")
            continue
        
        subject_info = {}
        subject_info["subject_id"] = subject_id
        subject_info["image_nifti_path"] = image_nifti_path
        subject_info["mask_nifti_path"] = mask_nifti_path

        subject_metadata = generate_subject_metadata(dataset_info, subject_info, additional_metadata)
        dataset_metadata[subject_id] = subject_metadata
        
    # Preprocess subjects to extract and normalize sex, age, and weight
    processed_subjects = preprocess_subjects(dataset_metadata)

    # Stratify and sample subjects based on the new balanced split strategy
    split_assignments = stratify_and_sample(processed_subjects, split_ratios, seed=42)

    # Update dataset metadata with split information
    for split, subjects in split_assignments.items():
        for subject in subjects:
            subject_id = subject['subject_id']
            if subject_id in dataset_metadata:
                dataset_metadata[subject_id]['Split'] = split

    return dataset_metadata

def create_slice_path_metadata(npy_image_dir:str, npy_mask_dir:str, dataset_info):
    """
    Function to create "slices metadata" for every subject in dataset. "Slices metadata" summarizes 
    file path information for all slices in the subject's volume.

    Params:
    - npy_image_dir: Folder containing NPY images.
    - npy_mask_dir: Folder containing NPY masks.
    - dataset_info: Additional characteristics from dataset.

    Returns:
    - subject_dataframes: Dictionary of dataframes with slice information (values) for every subject (keys)
    - subject_processed: Dictionary of whether the subject was processed (True/False values) for every subject (keys)
    """
    
    allowed_extensions = ['.npy']

    subject_dataframes = {}  # Dictionary to store DataFrames for each subject
    subject_processed = {}  # Dictionary to keep track of processed subjects

    # Create a list of subject ids
    subject_id_list = volume_ids_in_dir(npy_image_dir)

    # Create subject dataframe
    for subject_id in subject_id_list:
        # Determine paths to images and masks
        img_file_paths = volume_id_file_paths_in_dir(npy_image_dir, subject_id)
        img_file_paths = [f for f in img_file_paths if any([f.endswith(ext) for ext in allowed_extensions])]
        mask_file_paths = [f.replace(npy_image_dir, npy_mask_dir) for f in img_file_paths]
        # Verify files exist
        for mask_file in mask_file_paths:
            if not os.path.exists(mask_file) and os.path.isfile(mask_file):
                print(f"Files for subject {subject_id} not found.")
                subject_processed[subject_id] = False
                continue
        
        subject_dataframes[subject_id] = generate_slice_info_for_subject(img_file_paths, mask_file_paths, dataset_info)
        subject_processed[subject_id] = True
    
    return subject_dataframes, subject_processed


def metadata_creation(config_name, operation):

    cfg = load_dataset_config(config_name, root)

    if operation == 'A':
        dataset_info = {
            "csv_config": cfg.get("extra_metadata_cfg", {}),
            "dataset_name": cfg.get("metadata_cfg").get("dataset_name", "Unknown"),
            "anatomy": cfg.get("metadata_cfg").get("anatomy", "Unknown"),
        }
        
        dataset_metadata = create_metadata_for_statistics(
            cfg.get("mri_dicom_dir", ""),
            cfg.get("metadata_cfg").get("dicom_values", {}),
            dataset_info,
            cfg.get("dicom_column", ""),
            cfg.get("no_dicom_extension_flag", False),
        )
        save_metadata(dataset_metadata, cfg.get("stats_metadata_file"))
        
    elif operation == 'B':
        dataset_info = {
            "mask_labels": cfg.get("metadata_cfg").get("mask_labels", {}),
            "field_strength": cfg.get("metadata_cfg").get("field_strength", "Unknown"),
            "mri_sequence": cfg.get("metadata_cfg").get("mri_sequence", "Unknown"),
        }

        dataset_metadata = create_relevant_metadata_with_splits(
            cfg.get("nifti_image_dir", ""),
            cfg.get("nifti_mask_dir", ""),
            cfg.get("split_ratios", {'train': 0.7, 'val': 0.15, 'test': 0.15}),
            dataset_info,
            cfg.get("additional_metadata_file", None),
        )

        save_metadata(dataset_metadata, cfg.get("ml_metadata_file"))

    elif operation == 'C':

        dataset_info = {
            "npy_dir": cfg.get('npy_dir', ""),
            "dataset_name": cfg.get("metadata_cfg").get("dataset_name", "Unknown"),
        }
        subject_dataframes_dict, subject_processed = create_slice_path_metadata(
            os.path.join(cfg.get("npy_dir", ""), 'imgs'),
            os.path.join(cfg.get("npy_dir", ""), 'gts'),
            dataset_info,
        )

        for subject_id, subject_df in subject_dataframes_dict.items():
            output_path = os.path.join(cfg.get("slice_info_parquet_dir"), f'{subject_id}.parquet')
            save_metadata(subject_df, output_path)

    # -------------- Bonus metadata extraction ----------------- #
    elif operation == 'D':
        dataset_info = {
            "csv_config": cfg.get("extra_metadata_cfg", {}),
            "dataset_name": cfg.get("metadata_cfg").get("dataset_name", "Unknown"),
            "anatomy": cfg.get("metadata_cfg").get("anatomy", "Unknown"),
        }

        dicom_metadata, demographic_metadata, complete_dicom_metadata = create_metadata_for_acquisition_info(
            cfg.get("mri_dicom_dir", ""),
            cfg.get("metadata_cfg").get("dicom_values", {}),
            dataset_info,
            cfg.get("dicom_column", ""),
            cfg.get("no_dicom_extension_flag", False),
        )
        
        save_metadata(dicom_metadata, cfg.get("dicom_metadata_file").replace('.txt','.csv'))
        save_dicom_metadata_summary(complete_dicom_metadata, cfg.get("dicom_metadata_file"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert medical data to NIfTI format.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    parser.add_argument('--operation', type=str, required=True, choices=['A', 'B', 'C', 'D'], help='Select operation: A for statistics metadata, B for primary metadata with splits, C for slice parquet tables')

    args = parser.parse_args()
    config_name = args.config_name + '.yaml'

    metadata_creation(config_name, args.operation)