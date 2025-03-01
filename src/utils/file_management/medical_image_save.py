# src/utils/medical_image_save.py
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def save_nifti_itk(data, file_name):
    # Using SimpleITK to create and save a NIfTI image
    import SimpleITK as sitk
    # You may need to adjust the image parameters (e.g., direction, spacing) here if necessary.
    image = sitk.GetImageFromArray(data)
    return sitk.WriteImage(image, file_name)

def save_nifti(data, file_name, reference_nifti_path: str = None, save_method: str = "nibabel"):
    """
    Save data as a NIfTI file.
    
    If a reference_nifti_path is provided, its affine and header will be used.
    The save_method flag controls whether to use nibabel or SimpleITK for saving.
    """
    if save_method.lower() == "sitk":
        return save_nifti_itk(data, file_name)
    else:
        if reference_nifti_path is not None:
            # Use the reference file's affine and header.
            original_nii = nib.load(reference_nifti_path)
            new_nii = nib.Nifti1Image(data, original_nii.affine, original_nii.header)
        else:
            # Use an identity affine.
            new_nii = nib.Nifti1Image(data, np.eye(4))
        return nib.save(new_nii, file_name)

def save_prediction(data, save_dir, filename, output_ext, save_method="nibabel"):
    # Create output directory for predictions
    output_dir = os.path.join(save_dir, 'pred')
    os.makedirs(output_dir, exist_ok=True)
    
    # If saving as npz, compress and save the data
    if output_ext == 'npz':
        npz_output_path = os.path.join(output_dir, f"{filename}.npz")
        np.savez_compressed(npz_output_path, seg=data)
    else:
        # Depending on the save_method, choose the appropriate NIfTI saving function.
        if save_method.lower() == "sitk":
            save_nifti_itk(data, os.path.join(output_dir, f"{filename}.{output_ext}"))
        else:
            save_nifti(data, os.path.join(output_dir, f"{filename}.{output_ext}"))


def save_prediction_for_ITK(seg_3D, save_dir, filename, output_ext):
    """
    Save prediction as .nii.gz for ITK-SNAP compatibility - if original data source wasn't already simpleITK compatible.
    """
    output_dir = os.path.join(save_dir, 'pred')
    os.makedirs(output_dir, exist_ok=True)
    nifti_output_path = os.path.join(output_dir, f"{filename}_itk.nii.gz")
    seg_3D_transposed = np.transpose(seg_3D, (2, 1, 0))
    seg_3D_reversed = seg_3D_transposed[::-1]
    seg_3D_flipped = np.flip(seg_3D_reversed, axis=0)
    new_nii = nib.Nifti1Image(seg_3D_flipped, np.eye(4))
    nib.save(new_nii, nifti_output_path)
    return


# def save_nifti_nib(data, file_name):
#     new_nii = nib.Nifti1Image(data, np.eye(4))
#     return nib.save(new_nii, file_name)

# def save_nifti(data, file_name, reference_nifti_path: str = None):
#     """
#     Save data as a NIfTI file.
#     If a reference_nifti_path is provided, use its affine and header.
#     """
#     if reference_nifti_path is not None:
#         # Load the original NIfTI file to use as a template for affine/header
#         original_nii = nib.load(reference_nifti_path)
#         new_nii = nib.Nifti1Image(data, original_nii.affine, original_nii.header)
#     else:
#         # Create a new NIfTI image using an identity affine transformation matrix
#         new_nii = nib.Nifti1Image(data, np.eye(4))
#     return nib.save(new_nii, file_name)

# def save_prediction(seg_3D, save_dir, filename, output_ext):
#     """
#     Save prediction as .nii.gz or .npz (with key 'seg').
#     """
#     output_dir = os.path.join(save_dir, 'pred')
#     os.makedirs(output_dir, exist_ok=True)
#     if output_ext == 'npz':
#         npz_output_path = os.path.join(output_dir, f"{filename}.npz")
#         np.savez_compressed(npz_output_path, seg=seg_3D)
#     else:
#         nifti_output_path = os.path.join(output_dir, f"{filename}.nii.gz")
#         new_nii = nib.Nifti1Image(seg_3D, np.eye(4))
#         nib.save(new_nii, nifti_output_path)
#     return