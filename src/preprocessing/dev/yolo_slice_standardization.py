import os
import sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
join = os.path.join

from scipy.ndimage import label as scipy_label

import argparse

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.file_management.config_handler import load_dataset_config

# -------------------- DATA LOADING FUNCTIONS --------------------
from src.utils.file_management.file_handler import load_nifti

# ----------------- DATA TRANSFORMATION FUNCTIONS -----------------
from src.preprocessing.dev.yolo_prep import MaskPrep, ImagePrep, write_list_to_file, get_bounding_boxes

# --------------------- DATA SAVING FUNCTIONS ---------------------


def save_processed_data(image_data:np.ndarray, mask_data:np.ndarray, z_indices, output_dir:str, base_name):
    """
    Save processed image and mask data as .npy files after resizing and normalization.
    Params:
    - image_data: 3D slice data 
    - mask_data: 3D mask data 
    - output_dir: directory to save files
    - base_name: file name expects a unique volume id
    Output:
    - Saves 3 channel images slices in output_dir/imgs/base_name-###.npy, where ### is the slice index
    - Saves masks slices (1 channel) in output_dir/gt/base_name-###.npy, where ### is the slice index
    """
    images_dir = os.path.join(output_dir, "imgs") # "images"
    masks_dir = os.path.join(output_dir, "gts")   # "masks"
    annotations_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for i, (img_slice, mask_slice) in enumerate(zip(image_data, mask_data)):

        # yolo wants 3 channels, and its challenging dealing with a single channel in the yolo codebase
        img_slice_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)

        print(f"img_slice_3c dtype: {img_slice_3c.dtype}")

        image_file_name = f"{base_name}-{str(z_indices[i]).zfill(3)}"

        np.save(os.path.join(images_dir, image_file_name + ".npy"), img_slice_3c)  # wut their size is (640, 1024) to (640, 1024, 3)
        np.save(os.path.join(masks_dir, image_file_name + ".npy"), mask_slice)  # wut their size is (640, 1024)

        write_list_to_file(join(annotations_dir, image_file_name + ".txt"), get_bounding_boxes(mask_slice, instance=True))

        sys.stdout.flush()  # Ensure output is flushed immediately


# -------------------- MAIN PIPELINE FUNCTION --------------------
        
def nifti_to_npy(mask_dir, image_dir, dataMaskPrep, dataImagePrep, output_dir, file_suffix):
    """
    Code to convert standardized NIfTI to preprocessed NPY for input to SAM during fine tuning.
    Params:
    - mask_dir: Directory with mask files.
    - image_dir: Directory with image files whose names correspond to the filenames in mask_dir.
    - dataMaskPrep: Class with functions to preprocess masks for SAM
    - dataImagePrep: Class with functions to preprocess images for SAM
    - output_dir: Directory to save standardized npy files (output_dir/imgs/... and output_dir/gts/...)
    - file_suffix: Input file extension
    Outputs:
    - Saves 3 channel images slices in output_dir/imgs/base_name-###.npy, where ### is the slice index
    - Saves masks slices (1 channel) in output_dir/gt/base_name-###.npy, where ### is the slice index
    """
    # Process all files in the mask directory
    names=sorted(os.listdir(mask_dir))
    
    # Iterate through each file name provided in the 'names' list
    for name in tqdm(names, dynamic_ncols=True, position=0):

        # Load the mask data from the specified directory and file name
        mask_path = os.path.join(mask_dir, name)
        mask_data = np.uint8(load_nifti(mask_path))
        
        # Apply preprocessing steps to the mask data:
        prepped_mask_data, z_indices = dataMaskPrep.preprocess_mask(mask_data)
        
        # Load and process the corresponding image data:
        image_path = os.path.join(image_dir, name)
        image_data = load_nifti(image_path)
        
        prepped_image_data = dataImagePrep.preprocess_image(image_data, z_indices)

        # Derive the base filename for saving processed files without their original extension
        base_name = name.split(file_suffix)[0]

        # Save the processed image and mask data
        save_processed_data(prepped_image_data, prepped_mask_data, z_indices, output_dir, base_name)

def yolo_slice_standardization(config_name):
    """
    Parses config parameters to load standardized NIfTI images and masks, preprocess for YOLO, and save images and masks
    as slices in .npy files.
    """
    cfg = load_dataset_config(config_name, root)

    dataMaskPrep = MaskPrep(
        remove_label_ids=cfg.get('yolo_preprocessing_cfg').get('remove_label_ids'),
        target_label_id=cfg.get('yolo_preprocessing_cfg').get('target_label_id', None),  # Optional parameter with default
        voxel_threshold_3d=cfg.get('yolo_preprocessing_cfg').get('voxel_num_thre3d'),
        pixel_threshold_2d=cfg.get('yolo_preprocessing_cfg').get('voxel_num_thre2d'),
        image_size_tuple=cfg.get('yolo_preprocessing_cfg').get('image_size'),
        crop_non_zero_slices_flag = True,
        )
    
    dataImagePrep = ImagePrep(
        image_size_tuple=cfg.get('yolo_preprocessing_cfg').get('image_size'),
        make_square = cfg.get('preprocessing_cfg').get('make_square', False),
        )

    nifti_to_npy(
        mask_dir=cfg.get('nifti_mask_dir'),
        image_dir=cfg.get('nifti_image_dir'),
        dataMaskPrep=dataMaskPrep,
        dataImagePrep=dataImagePrep,
        output_dir=cfg.get('yolo_npy_dir'),
        file_suffix=".nii.gz"  
    )
    return


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess medical data for YOLO analysis.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()

    # Load the configuration file
    config_name = args.config_name + '.yaml'

    yolo_slice_standardization(config_name)