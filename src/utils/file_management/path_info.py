import os
import pandas as pd
import re
from typing import Union


def file_without_extension_from_path(file_name:str):
    """ Extract filenames without extension
    Caution: fails for files with periods but no extension (ex. dicom file: "1.2.345")
    """
    # Removes file extension
    # If zipped, remove zipped file extension (name.dcm.gz, name.nii.gz)
    if file_name.endswith('.gz'):
        file_name = file_name.rstrip('.gz')
    return os.path.splitext(file_name)[0] # Assuming file_name can be used as subject_id

def pair_files(image_path:str, mask_path:str, path_ext:str='', file_prefix:str=''):

    if os.path.isdir(image_path) and os.path.isdir(mask_path):
        if (path_ext) and (file_prefix):
            image_files = [f for f in os.listdir(image_path) if (f.startswith(file_prefix) and f.endswith(path_ext))]
        elif (path_ext):
            image_files = [f for f in os.listdir(image_path) if f.endswith(path_ext)]
        elif (file_prefix):
            image_files = [f for f in os.listdir(image_path) if f.startswith(file_prefix)]
        else:
            image_files = [os.listdir(image_path)]

        mask_files = set(os.listdir(mask_path))
        selected_pairs = [(os.path.join(image_path, f), os.path.join(mask_path, f))
                        for f in image_files if f in mask_files]
    else:
        selected_pairs = [(image_path, mask_path)]
    return selected_pairs

def pair_files_with_diff_ext(image_path:str, mask_path:str, image_ext:str='', mask_ext:str='',file_prefix:str=''):

    if os.path.isdir(image_path) and os.path.isdir(mask_path):
        if (image_ext) and (file_prefix):
            image_files = [f for f in os.listdir(image_path) if (f.startswith(file_prefix) and f.endswith(image_ext))]
        elif (image_ext):
            image_files = [f for f in os.listdir(image_path) if f.endswith(image_ext)]
        elif (file_prefix):
            image_files = [f for f in os.listdir(image_path) if f.startswith(file_prefix)]
        else:
            image_files = [os.listdir(image_path)]

        if (mask_ext):
            mask_files = [f for f in os.listdir(mask_path) if f.endswith(mask_ext)]
        else:
            mask_files = [os.listdir(mask_path)]

        selected_pairs = []
        for f_img in image_files:
            f_mask = f_img.rstrip(image_ext)+mask_ext
            if f_mask in mask_files:
                selected_pairs.append( (os.path.join(image_path, f_img), os.path.join(mask_path, f_mask)) )
    else:
        selected_pairs = [(image_path, mask_path)]
    return selected_pairs

def pair_files_in_split(image_dir:str, mask_dir:str, path_ext:str='', split_dict:dict={}, split_text:str=''):
    """Split and select"""
    selected_pairs = pair_files(image_dir, mask_dir, path_ext)
    
    # Check if the file is part of the test set if volume_metadata is provided
    if split_dict and split_text:
        valid_pairs = []
        for image_path, mask_path in selected_pairs:
            # Check whether file name without extension is assigned to the desired split
            file_name = os.path.basename(image_path)
            file_name_without_ext = file_without_extension_from_path(file_name)
            if split_dict.get(file_name_without_ext, {}).get('Split') == split_text:
                valid_pairs.append((image_path, mask_path))
            
    return valid_pairs

def extract_data_paths(data_dir:str, columns_to_process:Union[list[str], str, None] = None, subject_test_sample:int = None):
    """
    Reads data_dir to determine paths to data and the associated subject ids.
    Args:
        data_dir: Directory or csv of the original data.
        columns_to_process: Relevant csv columns names. List of strings or string.
        subject_test_sample (optional): number of subjects to return
    Returns:
        data_paths_info: tuple of tuples 
                        for a csv ((path to the data, subject id))
                        for a dir ((path to the data, filename without extension))
    """
    # Check whether data directory is valid
    assert os.path.exists(data_dir) and (data_dir.endswith('.csv') or os.path.isdir(data_dir)), \
            "Error: The path provided in 'data_dir' is neither a valid directory nor a CSV file."
    
    # Create a list of tuples tha specifies the path(s) to data and subject id 
    data_paths_info = []
    
    # Load files for subjects
    if data_dir.endswith('.csv'):
        subjects_df = pd.read_csv(data_dir)
        
        # Iterate over each row in the DataFrame, processing based on the configuration
        for _, row in subjects_df.iterrows():
            # Ensure column names are a list, converts single string to list
            if not isinstance(columns_to_process, list):
                columns_to_process = [columns_to_process]
            # Dynamically determine columns to process based on presence in the row
            data_paths = [row[column].strip() for column in columns_to_process \
                                if pd.notnull(row[column])]
            # Check for valid input
            if not isinstance(data_paths, list):
                raise ValueError('Invalid column name. Does not match .csv file.')
            subject_id = row['subject_id']
            
            data_paths_info.append((data_paths, subject_id))
    
    # Process each file or directory within the data_dir
    elif os.path.isdir(data_dir):
        for file_name in os.listdir(data_dir):
            data_paths = [os.path.join(data_dir, file_name)]
            # Assuming file_name can be used as subject_id, Removes file extension, 
            subject_id = file_without_extension_from_path(file_name) 

            data_paths_info.append((data_paths, subject_id))
    else:
        raise ValueError("The path provided is neither a valid directory nor a CSV file.")
    
    # Apply the slicing only if subject_test_sample is provided and is a positive integer
    # TODO - This should be moved to after the function is called 
    if subject_test_sample and subject_test_sample > 0:
        data_paths_info = data_paths_info[0:subject_test_sample]

    return data_paths_info


# ---------------- Npy Volume ID Path Functions ----------------

def volume_id_from_file(file_name:str):
    """ Extract the subject ID from the file name. """
    # Removes text specifying slice number and file extension.
    return file_name.rsplit('-',1)[0]

def volume_ids_in_dir(npy_files_dir:str):
    """ Extract unique subject IDs from all files in directory. """
    volume_id_list = []
    for file_name in os.listdir(npy_files_dir):
            volume_id = volume_id_from_file(file_name)
            if volume_id not in volume_id_list:
                volume_id_list.append(volume_id)
    return volume_id_list

def volume_id_file_paths_in_dir(npy_files_dir:str, volume_id:str):
    """ Extract all files with unique subject IDs in directory. """
    file_paths = []
    for f in sorted(os.listdir(npy_files_dir)):
        file_path = os.path.join(npy_files_dir,f)
        if f.startswith(volume_id) and os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths


def pair_volume_id_paths(image_path:str, mask_path:str):

    if os.path.isdir(image_path) and os.path.isdir(mask_path):
        image_ids = volume_ids_in_dir(image_path)
        mask_ids = volume_ids_in_dir(mask_path)

        selected_pairs = [(os.path.join(image_path, f), os.path.join(mask_path, f))
                        for f in image_ids if f in mask_ids]
    else:
        selected_pairs = []
    return selected_pairs


# ---------------- MISC ----------------
def alphanumeric_sort(strings):
    """
    Sort files by name without leading zeros. Supports 1, 2, ... 10, 11, instead of 1, 11, 2, 22, ...
    """
    def alphanumeric_key(string):
        return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', string)]
    return sorted(strings, key=alphanumeric_key)