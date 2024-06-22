import nibabel as nib
import json
import numpy as np

# -------------------- For Standardized NIfTI Data --------------------

def load_nifti(file_path:str) -> np.ndarray:
    """
    Load data from NIfTI file and return as a numpy array. 
    """
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return data

def save_nifti(data, file_name, reference_nifti_path:str=None):
    if reference_nifti_path is not None:
        # Load the original NIfTI file to use as a template for affine/header
        original_nii = nib.load(reference_nifti_path)
        # Create a new NIfTI image with the segmentation data
        new_nii = nib.Nifti1Image(data, original_nii.affine, original_nii.header)
    else:
        # Create a new NIfTI image using an identity affine transformation matrix 
        new_nii = nib.Nifti1Image(data, np.eye(4))
    return nib.save(new_nii, file_name)


# -------------------- For Metadata --------------------

def load_json(file_path:str):
    """ Load JSON file """
    with open(file_path, 'r') as file:
        info = json.load(file)
    return info

def save_json(data, output_path:str):
    """ Save the metadata to a JSON file """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    return True

def save_parquet(df, output_path:str):
    """ Save data to parquet file """
    df.to_parquet(output_path, index=False, compression='snappy')
    pass


# ------------- For Standardized NPY & NPZ Data -------------

def load_npy(filename:str, key:str=''):
    """
    Load data from a NumPy file using a specified key.
    """
    data = np.load(filename, "r", allow_pickle=True)
    if key:
        data = data[key]
    return data

def load_npz(filename:str, key:str=''):
    """
    Load data from a NumPy NPZ file using a specified key.
    """
    data = np.load(filename, "rb", allow_pickle=True)
    if key:
        data = data[key]
    return data