
# save_prediction
# save_prediction_for_ITK
# extract_filename
# determine_run_directory
# locate_files
# load_dcm

import os
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk

from ultralytics import YOLO, RTDETR

# -------------------- DATA SAVING FUNCTIONS -------------------- #

def save_prediction(seg_3D, save_dir, filename, output_ext):
    """Save prediction as .nii.gz, or .npz file with key 'seg'."""
    # Ensure the updated paths exist
    output_dir = os.path.join(save_dir, 'pred')
    os.makedirs(output_dir, exist_ok=True)

    # Assuming filename is already the base filename without the extension
    if output_ext == 'npz':
        npz_output_path = os.path.join(output_dir, f"{filename}.npz")
        np.savez_compressed(npz_output_path, seg=seg_3D) 
    else:
        nifti_output_path = os.path.join(output_dir, f"{filename}.nii.gz")
        # Create a new NIfTI image using an identity affine transformation matrix 
        new_nii = nib.Nifti1Image(seg_3D, np.eye(4))
        nib.save(new_nii, nifti_output_path)
    return

def save_prediction_for_ITK(seg_3D, save_dir, filename, output_ext):
    """Save prediction as .nii.gz using SimpleITK for .nii.gz files."""
    # Ensure the output directory exists
    output_dir = os.path.join(save_dir, 'pred')
    os.makedirs(output_dir, exist_ok=True)

    nifti_output_path = os.path.join(output_dir, f"{filename}_itk10.nii.gz")

    # Transpose the numpy array to get the desired dimensions (512, 256, 15)
    seg_3D_transposed = np.transpose(seg_3D, (2, 1, 0))

    # Reverse the order of slices
    seg_3D_reversed = seg_3D_transposed[::-1]

    # Flip along the y-axis to correct orientation for ITK-SNAP
    seg_3D_flipped = np.flip(seg_3D_reversed, axis=0)

    # gh_rev = seg_3D_flipped[:,:, ::-1]

    # Create a new NIfTI image using an identity affine transformation matrix 
    new_nii = nib.Nifti1Image(seg_3D_flipped, np.eye(4))

    nib.save(new_nii, nifti_output_path)
    
    return


def is_nifti_file(file_name):
    return file_name.endswith('.nii') or file_name.endswith('.nii.gz')

def locate_files(directory_or_file):
    if os.path.isfile(directory_or_file):
        # It's a single file (could be NIfTI or other file)
        return [directory_or_file]
    elif os.path.isdir(directory_or_file):
        contents = os.listdir(directory_or_file)
        full_paths = [os.path.join(directory_or_file, f) for f in contents]

        if all(os.path.isdir(p) for p in full_paths):
            # It's a folder of folders (likely DICOM folders)
            return full_paths
        elif all(is_nifti_file(f) for f in contents):
            # It's a folder of NIfTI files
            return full_paths
        else:
            # It's a single folder (likely a DICOM folder)
            return [directory_or_file]
    else:
        raise ValueError(f"{directory_or_file} is not a valid file or directory")


# from preprocessing - 
def load_dcm(dcm_dirpath: str):
    """
    Load DICOM images from a folder, ensuring slices are correctly ordered
    based on InstanceNumber from 1 onwards, with anatomically correct orientation.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names_unsorted = reader.GetGDCMSeriesFileNames(dcm_dirpath)

    # Function to extract InstanceNumber from DICOM metadata
    def get_instance_number(dcm_path):
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(dcm.InstanceNumber)

    # Step 1: Sort filenames based on InstanceNumber to ensure correct order
    dicom_names_sorted = sorted(dicom_names_unsorted, key=get_instance_number)

    # Load the volume
    reader.SetFileNames(dicom_names_sorted)
    image = reader.Execute()
    
    # Convert SimpleITK image to numpy array format
    image_array = sitk.GetArrayFromImage(image)

    return image_array


# ----------------------------------------------------------------------- #

def extract_filename(img_file: str):
    """ Extract filenames without extension
    Handles both files and directory paths.
    Caution: fails for files with periods but no extension (ex. dicom file: "1.2.345")
    """
    # Check if the path is a directory
    if os.path.isdir(img_file):
        # Use the folder name as the file name
        file_name = os.path.basename(os.path.normpath(img_file))
    else:
        file_name = os.path.basename(img_file)
        
        # Removes file extension
        # If zipped, remove zipped file extension (name.dcm.gz, name.nii.gz)
        if file_name.endswith('.gz'):
            file_name = file_name.rstrip('.gz')
        file_name = os.path.splitext(file_name)[0] # Assuming file_name can be used as subject_id

    return file_name


def determine_run_directory(base_dir, task_name, group_name=None):
    """
    Determines the next run directory for storing experiment data.
    """
    if group_name !=None:
        base_path = os.path.join(base_dir, task_name, group_name)
    else:
        base_path = os.path.join(base_dir, task_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Filter for directories that start with 'Run_' and are followed by an integer
    existing_runs = []
    for d in os.listdir(base_path):
        if d.startswith('Run_') and os.path.isdir(os.path.join(base_path, d)):
            parts = d.split('_')
            if len(parts) == 2 and parts[1].isdigit():  # Check if there is a number after 'Run_'
                existing_runs.append(d)
    
    if existing_runs:
        # Sort by the integer value of the part after 'Run_'
        existing_runs.sort(key=lambda x: int(x.split('_')[-1]))
        last_run_num = int(existing_runs[-1].split('_')[-1])
        next_run_num = last_run_num + 1
    else:
        next_run_num = 1
    
    run_directory = f'Run_{next_run_num}'
    full_run_path = os.path.join(base_path, run_directory)
    os.makedirs(full_run_path, exist_ok=True)
    
    return full_run_path
