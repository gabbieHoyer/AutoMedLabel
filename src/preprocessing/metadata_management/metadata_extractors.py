# Responsible for extracting and saving metadata from the medical images.
# Functions for reading metadata and saving it in a structured format (like JSON) would go here.

import os
import pyarrow
import pandas as pd
import numpy as np
import pydicom

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    dotenv=True, # load environment variables from .env if exists in root directory
)

from src.utils.file_management.file_loaders_savers import save_json, save_parquet

def extract_demographics_from_csv(volume_csv_config:dict):
    """
    Extract and standardize specified demographics information from a CSV file for a specific subject,
    based on a configuration and the subject_id.

    Parameters:
    - volume_csv_config: Configuration to extract volume metadata from CSV for a single subject.
        - csv_file_path: Path to the CSV file.
        - subject_id_col: Column name in the CSV file that contains subject identifiers.
        - column_mapping: A dictionary mapping standardized field names to actual column names in the CSV file. 
                Ex. {standard_name: actual_name}
        - current_subject_id: Unique id to identify relevant row in CSV that corresponds to dcm_file_path.

    Returns:
    - A dictionary with the specified, standardized columns as values for the given subject_id.
    """
    csv_file_path = volume_csv_config['csv_file_path']
    subject_id_col = volume_csv_config['subject_id_col']
    column_mapping = volume_csv_config['column_mapping']
    current_subject_id = volume_csv_config['subject_id'] 
    demographics_dict = {}

    if not os.path.exists(csv_file_path):
        print(f"Error: Demographics CSV file '{csv_file_path}' does not exist.")
        return demographics_dict

    demographic_data = pd.read_csv(csv_file_path)
    
    # Filter the DataFrame for the row matching the current subject_id
    subject_row = demographic_data[demographic_data[subject_id_col] == current_subject_id]
    
    if not subject_row.empty:
        # There should only be one matching row, so we use iloc[0] to access it
        for standard_name, actual_name in column_mapping.items():
            # Check if the actual column name exists in the DataFrame to avoid KeyError
            if actual_name in demographic_data.columns:
                # Store the data using the standardized name in the dictionary
                demographics_dict[standard_name] = subject_row.iloc[0][actual_name]
    else:
        print(f"Warning: No demographic data found for subject '{current_subject_id}'.")

    return demographics_dict

# ------------------------------------------------------------------------------------------

def extract_header_info(dcm_file_path:str, dicom_values:dict, volume_csv_config:dict=None):
    """
    Extract metadata from all files in the DICOM folder using the keys specified in the config. Add 
    demographic information as well, if available.

    Parameters:
    - dcm_file_path: Full path to dicom file
    - dicom_values: Dictionary of standardized and actual keys to metadata.
    - volume_csv_config (optional): Info to extract additional demographic info from CSV. If information is 
            provided in both dicom header and CSV, value will be extracted from the dicom header.

    Returns:
    - metadata: Metadata from dicom header
    - demographics: Demographic info from dicom header and additional csv file.
    """

    def convert_to_native_type(value):
        """Convert numpy data types and containers of numpy data types to Python native data types for JSON serialization."""
        if isinstance(value, np.number):
            return value.item()  # Convert numpy numbers to Python scalars
        elif isinstance(value, np.generic):
            return value.item()  # Convert other numpy generic types
        elif isinstance(value, list):
            return [convert_to_native_type(v) for v in value]  # Recursively convert lists
        return value

    def convert_to_list(value):
        """Ensure multi-valued DICOM attributes or numpy arrays are converted to lists of native Python types."""
        if isinstance(value, pydicom.multival.MultiValue) or isinstance(value, np.ndarray):
            return [convert_to_native_type(v) for v in value]
        else:
            return convert_to_native_type(value)

    ds = pydicom.dcmread(dcm_file_path)
    metadata = {}
    demographics = {}
    demographic_keys = ['Age', 'Sex', 'Weight', 'BMI', 'Race']  # Specify demographic keys

    # Extract DICOM values based on keys specified in the config
    for key, field_name in dicom_values.items():
        #import pdb; pdb.set_trace()
        if hasattr(ds, field_name):
            value = getattr(ds, field_name)
            value = convert_to_list(value)
            # Determine if the key should go into demographics or metadata
            if key in demographic_keys:
                demographics[key] = value
            else:
                metadata[key] = value
    
    if volume_csv_config:
        csv_demographics = extract_demographics_from_csv(volume_csv_config)
        
        # Merge all demographics from CSV, supplementing missing ones and adding additional demographics
        for key, value in csv_demographics.items():
            # Use demographics from CSV, if not present in DICOM or if the value in DICOM is empty
            if key in demographic_keys and (key not in demographics or not demographics[key]):
                demographics[key] = convert_to_native_type(value)

    #TODO - Warning, not all demographic keys wil be in "demographics" variable. Gabbie, if this is okay, delete this comment ;)
    
    return metadata, demographics  # Return demographics as a separate dictionary


# ------------------------------------------------------------------------------------------

# def extract_dicom_data(file_path):
#     ds = pydicom.dcmread(file_path)
#     return ds

def summarize_unique_dicom_data(dicom_metadata):
    summary_dicom_metadata = {}
    for col_name in dicom_metadata.columns:
        if col_name in ['PID', 'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID']:
            continue
        print(col_name)
        try:
            unique_vals_counts = dicom_metadata.loc[:,col_name].value_counts().to_dict()
        except:
            unique_vals_counts = dicom_metadata.loc[:,col_name].apply(tuple).value_counts().to_dict()
        print(unique_vals_counts)
        unique_vals = dicom_metadata.loc[:,col_name].drop_duplicates().tolist()
        try:
            print('min - max')
            print(min(unique_vals), max(unique_vals))
            print()
        except:
            pass
        summary_dicom_metadata['unique_'+col_name] = unique_vals_counts
    return summary_dicom_metadata

# ------------------------------------------------------------------------------------------

def generate_subject_metadata(dataset_info:dict, subject_info:dict, additional_metadata):
    """
    Generate metadata relevant to ML/AI data splitting (train/val/test) for a single subject.

    Parameters:
    - dataset_info: Additional characteristics from dataset.
    - subject_info: Additional characteristics from dataset.
    - additional_metadata: #TODO Dictionary of all subject metadata.

    Returns:
    - subject_metadata: Dictionary with a subset of info from "additional_metadata" for a single subject.
    """
    subject_id = subject_info["subject_id"]

    subject_metadata =  {
        'subject_id': subject_id,
        'image_nifti': subject_info["image_nifti_path"],
        'mask_nifti': subject_info["mask_nifti_path"]
    }

    # Extract specific information from additional_metadata
    if subject_id in additional_metadata:
        demographics = additional_metadata[subject_id].get('Demographics', {})
        subject_metadata['Sex'] = demographics.get('Sex', 'Unknown')
        subject_metadata['Age'] = demographics.get('Age', 'Unknown')
        subject_metadata['Weight'] = demographics.get('Weight', 'Unknown')
        subject_metadata['Dataset'] = additional_metadata[subject_id].get('Dataset', 'Unknown')
        subject_metadata['Anatomy'] = additional_metadata[subject_id].get('Anatomy', 'Unknown')
        subject_metadata['num_slices'] = len(additional_metadata.get(subject_id, {}).get('slices', {}))

    # Add additional characteristics from dataset
    subject_metadata['mask_labels'] = dataset_info["mask_labels"]
    subject_metadata['field_strength'] = dataset_info["field_strength"]
    subject_metadata['mri_sequence'] = dataset_info["mri_sequence"]

    return subject_metadata

# ------------------------------------------------------------------------------------------

# def generate_slice_info_for_subject(img_file_paths, mask_file_paths, dataset_info):
def generate_slice_info_for_subject(img_file_paths, mask256_file_paths, label_file_paths, dataset_info):  #mask_file_paths,
    """
    For a single subject/volume, summarize file path info about all slices.

    Parameters:
    - img_file_paths: List of valid paths to all images that correspond to one subject/volume.
    - mask_file_paths: List of valid paths to all masks that correspond to one subject/volume.
    - label_file_paths: List of valid paths to all labels that correspond to one subject/volume (optional).
    - dataset_info: Additional characteristics from dataset.

    Returns:
    - subject_df: Summary of all data paths and relevant slice info for one subject/volume.
    """
    data = []

    for img_path in sorted(img_file_paths):
        subject_id = os.path.basename(img_path).rsplit('-',1)[0]
        slice_number = img_path.rsplit('-')[-1].split('.')[0].zfill(3)  # Format slice_number with leading zeros
        # mask_path = [f for f in mask_file_paths if f"{subject_id}-{slice_number}" in f][0]
        mask256_path = [f for f in mask256_file_paths if f"{subject_id}-{slice_number}" in f][0]

        if label_file_paths is not None:
            label_path = [f for f in label_file_paths if f"{subject_id}-{slice_number}" in f][0]
            data.append({'subject_id': subject_id,
                            'slice_number': slice_number,
                            'npy_base_dir': dataset_info['npy_dir'],
                            'npy_image_path': img_path,
                            # 'npy_mask_path': mask_path,
                            'npy256_mask_path': mask256_path,
                            'txt_label_path': label_path,
                            'Dataset': dataset_info['dataset_name'],
                            'mask_labels': 'placeholder'})
        else:
            data.append({'subject_id': subject_id,
                            'slice_number': slice_number,
                            'npy_base_dir': dataset_info['npy_dir'],
                            'npy_image_path': img_path,
                            # 'npy_mask_path': mask_path,
                            'npy256_mask_path': mask256_path,
                            'Dataset': dataset_info['dataset_name'],
                            'mask_labels': 'placeholder'})

    # subject_df = pd.concat([pd.DataFrame(data)])
    subject_df = pd.DataFrame(data)

    return subject_df




