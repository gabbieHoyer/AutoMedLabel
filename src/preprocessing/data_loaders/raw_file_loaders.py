"""
Script to load and save files with various extensions.
"""
import numpy as np
import h5py
from scipy.io import loadmat
import SimpleITK as sitk
import nibabel as nib
import pydicom

# ----------------------- RAW DATA lOADING -----------------------

"""
Module with functions to load data that was saved in various formats. 
"""
def load_npy(filename:str, key:str):
    """
    Load data from a NumPy file using a specified key.
    """
    if (key):
        data = np.load(filename, "r", allow_pickle=True)[key]
    return data

def load_npz(filename:str, key:str):
    """
    Load data from a NumPy NPZ file using a specified key.
    """
    with np.load(filename, "rb", allow_pickle=True) as npz_file:
        data = npz_file[key]
    return data

def load_h5(filename:str, key:str, data_type=np.float32):
    """
    Load data from an HDF5 file using a specified key.
    """
    with h5py.File(filename, 'r') as f:
        data = f[key][()].astype(data_type) 
    return data

def load_h5_keys(filename:str):
    f = h5py.File(filename, 'r')
    print(f'Keys: {f.keys()}')
    f.close()
    return

def load_int2(filename:str, dim_x=256, dim_y=256):
    """
    Load data from a int2 file and returns them with dimensions (256 x 256 x number of slices).
    """
    data_raw = np.fromfile(filename, dtype='>i2')
    data = data_raw.reshape((dim_x,dim_y,-1),order='F')
    #data = np.rot90(data, axes=(1,0))
    #data = np.flip(data, axis=1)
    return data  

def load_mat(filename:str, key:str, struct_as_record:bool=False):  # struct True
    """
    Load data from a .mat file
    Args:
        struct_as_record: default when using loadmat is true, false when loading imorphics
    """
    data = loadmat(filename, struct_as_record=struct_as_record)[key]
    return data

def load_mhd(filename:str):
    """
    Load data from a .mdh file
    """
    # Load the .mhd file using SimpleITK
    img = sitk.ReadImage(filename)
    # Convert to NIfTI format
    data = sitk.GetArrayFromImage(img)
    return data

def load_nifti(file_path:str) -> np.ndarray:
    """
    Load data from NIfTI file and return as a numpy array. 
    """
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return data


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


# def load_dcm(dcm_dirpath:str):
#     """
#     Load data from dicom folder using sitk.
#     """
#     # Read the DICOM series
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dcm_dirpath)

#     if not dicom_names:
#         raise ValueError("No DICOM files found in the specified folder.")

#     reader.SetFileNames(dicom_names)
#     image = reader.Execute()
#     # Convert SimpleITK image to np.array format
#     image = sitk.GetArrayFromImage(image)
#     return image

