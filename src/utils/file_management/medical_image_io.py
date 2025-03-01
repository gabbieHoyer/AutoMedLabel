# src/utils/medical_image_io.py
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom
import h5py
from scipy.io import loadmat

from .path_info import volume_id_file_paths_in_dir

# --- Additional functions originally in raw_file_loaders.py --- #

def load_npy(filename: str, key: str = '') -> np.ndarray:
    """
    Load data from a NumPy file using a specified key.
    Previous name: load_npy (from raw_file_loaders.py)
    """
    data = np.load(filename, "r", allow_pickle=True)
    if key:
        data = data[key]
    return data

def load_npz(filename: str, key: str = '') -> np.ndarray:
    """
    Load data from a NumPy NPZ file using a specified key.
    Previous name: load_npz (from raw_file_loaders.py)
    """
    with np.load(filename, "rb", allow_pickle=True) as npz_file:
        data = npz_file[key] if key else npz_file
    return data

def load_h5(filename: str, key: str, data_type=np.float32) -> np.ndarray:
    """
    Load data from an HDF5 file using a specified key.
    Previous name: load_h5 (from raw_file_loaders.py)
    """
    with h5py.File(filename, 'r') as f:
        data = f[key][()].astype(data_type)
    return data

def load_h5_keys(filename: str):
    """
    Print keys available in an HDF5 file.
    Previous name: load_h5_keys (from raw_file_loaders.py)
    """
    with h5py.File(filename, 'r') as f:
        print(f'Keys: {list(f.keys())}')
    return

def load_int2(filename: str, dim_x: int = 256, dim_y: int = 256) -> np.ndarray:
    """
    Load data from an int2 file and return them with dimensions (dim_x, dim_y, number of slices).
    Previous name: load_int2 (from raw_file_loaders.py)
    """
    data_raw = np.fromfile(filename, dtype='>i2')
    data = data_raw.reshape((dim_x, dim_y, -1), order='F')
    return data

def load_mat(filename: str, key: str, struct_as_record: bool = False) -> np.ndarray:
    """
    Load data from a .mat file using a specified key.
    Previous name: load_mat (from raw_file_loaders.py)
    """
    data = loadmat(filename, struct_as_record=struct_as_record)[key]
    return data

def load_mhd(filename: str) -> np.ndarray:
    """
    Load data from a .mhd file.
    Previous name: load_mhd (from raw_file_loaders.py)
    """
    img = sitk.ReadImage(filename)
    return sitk.GetArrayFromImage(img)

def load_dcm(dcm_dirpath: str) -> np.ndarray:
    """
    Load DICOM images from a folder, ensuring slices are correctly ordered.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names_unsorted = reader.GetGDCMSeriesFileNames(dcm_dirpath)

    def get_instance_number(dcm_path):
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(dcm.InstanceNumber)

    dicom_names_sorted = sorted(dicom_names_unsorted, key=get_instance_number)
    reader.SetFileNames(dicom_names_sorted)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)

def is_nifti_file(file_name: str) -> bool:
    return file_name.endswith('.nii') or file_name.endswith('.nii.gz')

def load_nifti(file_path: str) -> np.ndarray:
    """
    Load data from a NIfTI file and return as a NumPy array.
    """
    nifti = nib.load(file_path)
    return nifti.get_fdata()

def load_nifti_sitk(file_path: str) -> np.ndarray:
    """
    Load a NIfTI file using SimpleITK (for ITK-SNAP compatibility).
    """
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)

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

def load_data(file_path: str, key: str = '', use_sitk: bool = False) -> np.ndarray:
    """
    Load data from a .npz, .npy, or NIfTI file.
    
    This function supports:
      - .npz files: returns the array corresponding to the provided key (if any)
      - .npy files: returns the array (and if a key is provided, indexes into it)
      - .nii/.nii.gz files: returns the image data as a NumPy array;
          if use_sitk is True, uses SimpleITK for loading.
    """
    if file_path.endswith('.npz'):
        return load_npz(file_path, key=key)
    elif file_path.endswith('.npy'):
        return load_npy(file_path, key=key)
    elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        if use_sitk:
            return load_nifti_sitk(file_path)
        else:
            return load_nifti(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
def locate_files(directory_or_file: str) -> list:
    """
    Locate files or directories based on the input path.
    """
    if os.path.isfile(directory_or_file):
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

