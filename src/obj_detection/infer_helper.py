

import os
import random
from tqdm import tqdm
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
import pydicom

import SimpleITK as sitk

from ultralytics import YOLO, RTDETR

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

# from src.preprocessing.dev.sam_prep import MaskPrep, ImagePrep
from src.preprocessing.model_prep import MaskPrep, ImagePrep
from src.utils.file_management.file_handler import load_data
from src.finetuning.engine.models.sam import finetunedSAM

# ------------------- Visualization Tools ----------------------------------- #

def map_labels_to_colors(pred_mask, mask_labels):
    # Define a colormap that can provide a distinct color for each class
    color_map = plt.get_cmap('rainbow', len(mask_labels) - 1)  # Exclude background

    # Create an empty RGBA image
    colored_mask = np.zeros((*pred_mask.shape, 4), dtype=np.float32)  # Initialize with zeros

    # Map each label to a consistent color from the colormap
    for label_value, label_name in mask_labels.items():
        if label_value == 0:  # Skip the background
            continue
        mask = (pred_mask == label_value)
        color = color_map((label_value - 1) / (len(mask_labels) - 1))  # Get consistent RGBA color
        colored_mask[mask] = color  # Apply color where the label matches

    return colored_mask


def add_colorbar(fig, ax, mask_labels):
    # Get unique labels excluding background
    unique_labels = [label for label in mask_labels if label != 0]

    # Create a colormap based on unique labels
    base_cmap = plt.get_cmap('rainbow', len(unique_labels))
    colors = base_cmap(np.linspace(0, 1, len(unique_labels)))
    new_cmap = ListedColormap(colors)

    # Normalize according to the number of unique labels
    norm = Normalize(vmin=min(unique_labels), vmax=max(unique_labels))

    # Create a scalar mappable for colormap and normalization
    sm = ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])  # dummy array

    # Add colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation='vertical', aspect=10)
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([mask_labels[label] for label in unique_labels])

def visualize_full_pred(image, pred_mask, mask_labels, image_name, model_save_path):  
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with three subplots

    # Original image
    ax[0].imshow(image, cmap='gray')  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Prediction overlay
    ax[1].imshow(image, cmap='gray')  # Original image
    colored_pred_mask = map_labels_to_colors(pred_mask, mask_labels)
    im = ax[1].imshow(colored_pred_mask, alpha=0.5)  # Overlay the colored mask
    ax[1].set_title('Prediction Overlay')
    ax[1].axis('off')

    # Add colorbar
    add_colorbar(fig, ax, mask_labels)

    figure_file_path = os.path.join(model_save_path, "QC", f"{image_name}_full_pred.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory


# -----------------------------------------------------------------
def visualize_input(image, image_name, model_save_path):
    # Check the image shape and permute if necessary
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig = plt.figure(figsize=(10, 10))  # Create a figure with three subplots

    # Assuming image is in [C, H, W] format and is an RGB image
    plt.imshow(image, cmap='gray')  # Assuming image is in [C, H, W] format
    plt.title('Original Image')
    plt.axis('off')

    figure_file_path = os.path.join(model_save_path, "QC", f"{image_name}_yolo_input_QC.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path) 
    plt.close(fig)  # Close the figure to free memory

# -----------------------------------------------------------------

def visualize_pred(image, pred_mask, binary_pred, boxes, image_name, model_save_path):  # pred_mask_bin, label_id,
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with three subplots

    # Original image
    # Assuming image is in [C, H, W] format and is an RGB image
    ax[0].imshow(image)  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Prediction overlay
    ax[1].imshow(image)  # Original image
    ax[1].imshow(pred_mask, alpha=0.5)  # Overlay the colored mask
    ax[1].set_title('Prediction Overlay')
    ax[1].axis('off')

    # Prediction overlay
    ax[2].imshow(image)  # Original image
    ax[2].imshow(binary_pred, alpha=0.5)  # Overlay the colored mask
    ax[2].set_title('Binary Prediction Overlay')
    ax[2].axis('off')

    # Ensure boxes are in the correct format (list of numpy arrays)
    if not isinstance(boxes, list):
        boxes = [boxes]

    # Draw bounding boxes on the original image
    for box in boxes:
        # Convert box tensor to numpy array if it's not already
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        
    figure_file_path = os.path.join(model_save_path, "label_QC", f"{image_name}_pred.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory


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


# def save_prediction_for_ITK(seg_3D, save_dir, filename, output_ext):
#     """Save prediction as .nii.gz using SimpleITK for .nii.gz files."""
#     # Ensure the output directory exists
#     output_dir = os.path.join(save_dir, 'pred')
#     os.makedirs(output_dir, exist_ok=True)

#     nifti_output_path = os.path.join(output_dir, f"{filename}_itk10.nii.gz")

#     # Transpose the numpy array to get the desired dimensions (512, 256, 15)
#     seg_3D_transposed = np.transpose(seg_3D, (2, 1, 0))

#     # Reverse the order of slices
#     seg_3D_reversed = seg_3D_transposed[::-1]

#     # Flip along the y-axis to correct orientation for ITK-SNAP
#     seg_3D_flipped = np.flip(seg_3D_reversed, axis=0)

#     gh_rev = seg_3D_flipped[:,:, ::-1]

#     # Create a new NIfTI image using an identity affine transformation matrix 
#     new_nii = nib.Nifti1Image(gh_rev, np.eye(4))

#     nib.save(new_nii, nifti_output_path)
    
#     return

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

# -------------------- MODEL FUNCTIONS -------------------- #
def make_predictor(model_type:str, comp_cfg, initial_weights:str, finetuned_weights:str=None, device:str="cpu"):

    sam_model = sam_model_registry[model_type](checkpoint=initial_weights).to(device)

    # Model finetuning setup
    finetuned_model = finetunedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        config=comp_cfg
    ).to(device)

    # Check if finetuned_weights are the same as initial_weights
    if finetuned_weights == initial_weights:
        return finetuned_model

    # Check if a checkpoint exists to resume training
    if finetuned_weights and os.path.isfile(finetuned_weights):
        try:
            checkpoint = torch.load(finetuned_weights, map_location=device)
            finetuned_model.load_state_dict(checkpoint["model"])
        except Exception as e:
            # Decide whether to continue with training from scratch or to abort
            raise e
        
    # predictor = SamPredictor(sam_model)
    return finetuned_model

def make_prediction(predictor, img_slice, bbox):
    """Predict segmentation for image (2D) from bounding box prompt"""
    # img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)
    
    # predictor.set_image(img_slice) # Tensor object has no attribute 'astype'
    predictor.set_image(img_slice.astype(np.uint8))

    sam_mask, _, _ = predictor.predict(point_coords=None, 
                                       point_labels=None, 
                                       box=bbox[None, :], 
                                       multimask_output=False)
    return sam_mask

# -------------------- DATA POST-PROCESS FUNCTIONS -------------------- #
def postprocess_resize(mask, image_size_tuple:tuple[int,int], make_square):
    """Resize mask to new dimensions."""
    predMaskPrep = MaskPrep(make_square=make_square)
    resized_mask = predMaskPrep.resize_mask(mask_data = mask.astype(np.uint8),
                                            image_size_tuple = image_size_tuple)
    return resized_mask

def resize_prediction(sam_pred, image_size_tuple:tuple[int,int], label_id:int, make_square):
    """Convert SAM prediction into segmentation mask with original image dims and label ids."""
    sam_mask_resized = postprocess_resize(sam_pred, image_size_tuple, make_square)
    sam_mask = np.zeros_like(sam_mask_resized, dtype=np.uint8)
    sam_mask[sam_mask_resized > 0] = label_id
    return sam_mask


# def postprocess_resize(mask, image_size_tuple:tuple[int,int]):
#     """Resize mask to new dimensions."""
#     predMaskPrep = MaskPrep()
#     resized_mask = predMaskPrep.resize_mask(mask_data = mask.astype(np.uint8),
#                                             image_size_tuple = image_size_tuple)
#     return resized_mask

# def resize_prediction(sam_pred, image_size_tuple:tuple[int,int], label_id:int):
#     """Convert SAM prediction into segmentation mask with original image dims and label ids."""
#     sam_mask_resized = postprocess_resize(sam_pred, image_size_tuple)
#     sam_mask = np.zeros_like(sam_mask_resized, dtype=np.uint8)
#     sam_mask[sam_mask_resized > 0] = label_id
#     return sam_mask


def postprocess_prediction(sam_pred):
    # Ensure sam_pred is in the range [0, 255] and of type np.uint8
    sam_pred = sam_pred.astype(np.uint8) * 255  # Convert binary 0/1 to grayscale 0/255
    
    # Apply Gaussian blur
    smooth_mask = cv2.GaussianBlur(sam_pred, (5, 5), 0)
    
    # Apply binary thresholding to convert back to binary mask
    # _, processed_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)
    _, processed_mask = cv2.threshold(smooth_mask, 127, 1, cv2.THRESH_BINARY)
    
    # Convert to np.uint8 if necessary
    processed_mask = processed_mask.astype(np.uint8)
    
    return processed_mask

# --------------------------------------------------------------------------

def load_config(config_file_name, base_dir):
    config_path = os.path.join(base_dir, "config/obj_detection/inference", config_file_name)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    return config

def load_model(config):
    model_type = config.get('model_type', 'YOLO').upper()  # Default to YOLO if not specified

    if model_type == 'YOLO':
        model_class = YOLO
    elif model_type == 'RTDETR':
        model_class = RTDETR
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model_class

# def locate_files(directory_or_file):
#     if os.path.isfile(directory_or_file):
#         return [directory_or_file]
#     elif os.path.isdir(directory_or_file):
#         return [os.path.join(directory_or_file, f) for f in os.listdir(directory_or_file) if os.path.isfile(os.path.join(directory_or_file, f))]
#     else:
#         raise ValueError(f"{directory_or_file} is not a valid file or directory")

# def locate_files(directory_or_file):
#     if os.path.isfile(directory_or_file):
#         return [directory_or_file]
#     elif os.path.isdir(directory_or_file):
#         return [directory_or_file]
#     # [os.path.join(directory_or_file, f) for f in os.listdir(directory_or_file) if os.path.isfile(os.path.join(directory_or_file, f))]
#     else:
#         raise ValueError(f"{directory_or_file} is not a valid file or directory")


# def locate_files(directory_or_file):
#     if os.path.isfile(directory_or_file):
#         # It's a single file (could be NIfTI or other file)
#         return [directory_or_file]
#     elif os.path.isdir(directory_or_file):
#         contents = os.listdir(directory_or_file)
#         full_paths = [os.path.join(directory_or_file, f) for f in contents]
        
#         if all(os.path.isdir(p) for p in full_paths):
#             # It's a folder of folders (likely DICOM folders)
#             return full_paths
#         else:
#             # It's a single folder (could be NIfTI files or a single DICOM folder)
#             # return [directory_or_file]
#             return [os.path.join(directory_or_file, f) for f in os.listdir(directory_or_file) if os.path.isfile(os.path.join(directory_or_file, f))]
#     else:
#         raise ValueError(f"{directory_or_file} is not a valid file or directory")


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

# def extract_filename(img_file:str):
#     """ Extract filenames without extension
#     Caution: fails for files with periods but no extension (ex. dicom file: "1.2.345")
#     """
#     file_name = os.path.basename(img_file)

#     # Removes file extension
#     # If zipped, remove zipped file extension (name.dcm.gz, name.nii.gz)
#     if file_name.endswith('.gz'):
#         file_name = file_name.rstrip('.gz')
#     return os.path.splitext(file_name)[0] # Assuming file_name can be used as subject_id


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

# ----------------------------------------------------------------------------------- #
