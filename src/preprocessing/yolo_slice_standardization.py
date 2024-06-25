import os
import sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

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
from src.preprocessing.yolo_prep import MaskPrep, ImagePrep

# --------------------- DATA SAVING FUNCTIONS ---------------------
# def save_processed_data(image_data:np.ndarray, mask_data:np.ndarray, z_indices, output_dir:str, base_name):
#     """
#     Save processed image and mask data as .npy files after resizing and normalization.
#     Params:
#     - image_data: 3D slice data 
#     - mask_data: 3D mask data 
#     - output_dir: directory to save files
#     - base_name: file name expects a unique volume id
#     Output:
#     - Saves 3 channel images slices in output_dir/imgs/base_name-###.npy, where ### is the slice index
#     - Saves masks slices (1 channel) in output_dir/gt/base_name-###.npy, where ### is the slice index
#     """
#     images_dir = os.path.join(output_dir, "images")
#     masks_dir = os.path.join(output_dir, "masks")
#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(masks_dir, exist_ok=True)
    
#     for i, (img_slice, mask_slice) in enumerate(zip(image_data, mask_data)):
#         image_file_name = f"{base_name}-{str(z_indices[i]).zfill(3)}.npy"
#         np.save(os.path.join(images_dir, image_file_name), img_slice)
#         np.save(os.path.join(masks_dir, image_file_name), mask_slice)

#         sys.stdout.flush()  # Ensure output is flushed immediately

join = os.path.join

# def find_bounding_boxes(multiclass_mask):
#     # Find bounding boxes from segmentations for YOLO
#     # https://www.picsellia.com/post/how-to-train-yolov8-on-a-custom-dataset#4-The-YOLO-format
    
#     # Find unique labels in the image
#     unique_labels = np.unique(multiclass_mask)
#     #print('slice labels', len(unique_labels), unique_labels)
#     bounding_boxes = []

#     for label in unique_labels:
#         # Skip background label
#         if label == 0:
#             continue  
            
#         # Find coordinates of the current label
#         indices = np.where(multiclass_mask == label)

#         # Calculate bounding box coordinates
#         min_row, min_col = np.min(indices[0]), np.min(indices[1])
#         max_row, max_col = np.max(indices[0]), np.max(indices[1])

#         # Calculate center and dimensions
#         center_x = (min_col + max_col) / 2
#         center_y = (min_row + max_row) / 2
#         width = max_col - min_col
#         height = max_row - min_row

#         # Normalize
#         sz_x, sz_y = np.shape(multiclass_mask)
        
#         bounding_boxes.append(
#             str(label) +
#             ' ' + str(center_x/sz_x) +
#             ' ' + str(center_y/sz_y) +
#             ' ' + str(width/sz_x) +
#             ' ' + str(height/sz_y) )

#     return bounding_boxes





def find_bounding_boxes(multiclass_mask, instance=False):
    # Find bounding boxes from segmentations for YOLO
    # https://www.picsellia.com/post/how-to-train-yolov8-on-a-custom-dataset#4-The-YOLO-format
    
    # Find unique labels in the image
    unique_labels = np.unique(multiclass_mask)
    bounding_boxes = []

    for label in unique_labels:
        # Skip background label
        if label == 0:
            continue  
        
        if instance:
            # Process each instance separately
            gt2D = np.uint8(multiclass_mask == label)  # Binary mask for chosen class
            labeled_array, num_features = scipy_label(gt2D)

            for component in range(1, num_features + 1):
                component_mask = labeled_array == component
                y_indices, x_indices = np.where(component_mask)

                # Compute the bounding box for the selected component
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # Calculate center and dimensions
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # Normalize
                height_sz, width_sz = multiclass_mask.shape
                bounding_boxes.append(
                    str(label) +
                    ' ' + str(center_x / width_sz) +
                    ' ' + str(center_y / height_sz) +
                    ' ' + str(width / width_sz) +
                    ' ' + str(height / height_sz) )
        else:
            # Original logic for bounding box of entire label
            indices = np.where(multiclass_mask == label)

            # Calculate bounding box coordinates
            min_row, min_col = np.min(indices[0]), np.min(indices[1])
            max_row, max_col = np.max(indices[0]), np.max(indices[1])

            # Calculate center and dimensions
            center_x = (min_col + max_col) / 2
            center_y = (min_row + max_row) / 2
            width = max_col - min_col
            height = max_row - min_row

            # Normalize
            height_sz, width_sz = multiclass_mask.shape
            bounding_boxes.append(
                str(label) +
                ' ' + str(center_x / width_sz) +
                ' ' + str(center_y / height_sz) +
                ' ' + str(width / width_sz) +
                ' ' + str(height / height_sz) )

    return bounding_boxes


def write_list_to_file(file_path, data_list):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(str(item) + '\n')

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
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    annotations_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # import pdb; pdb.set_trace()
    
    for i, (img_slice, mask_slice) in enumerate(zip(image_data, mask_data)):

        # yolo wants 3 channels, and its significantly challenging dealing with a single channel in the yolo codebase

        # import pdb; pdb.set_trace()

        img_slice_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)

        # if npy_save:
        # image_file_name = f"{base_name}-{str(z_indices[i]).zfill(3)}.npy"

        # np.save(os.path.join(images_dir, image_file_name), img_slice_3c)  # wut their size is (640, 1024) to (640, 1024, 3)
        # np.save(os.path.join(masks_dir, image_file_name), mask_slice)  # wut their size is (640, 1024)

        image_file_name = f"{base_name}-{str(z_indices[i]).zfill(3)}"

        np.save(os.path.join(images_dir, image_file_name + ".npy"), img_slice_3c)  # wut their size is (640, 1024) to (640, 1024, 3)
        np.save(os.path.join(masks_dir, image_file_name + ".npy"), mask_slice)  # wut their size is (640, 1024)

        # else: # png save
        # image_file_name = f"{base_name}-{str(z_indices[i]).zfill(3)}"
        # plt.imsave(os.path.join(images_dir, image_file_name + ".png"), img_slice_3c)  # wut their size is (640, 1024) to (640, 1024, 3)
        # plt.imsave(os.path.join(masks_dir, image_file_name + ".png"), mask_slice)  # wut their size is (640, 1024)

        write_list_to_file(join(annotations_dir, image_file_name + ".txt"), find_bounding_boxes(mask_slice, instance=True))

        # import pdb; pdb.set_trace()

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
        #print(f"Processing {name}")
        
        # import pdb; pdb.set_trace()

        # Load the mask data from the specified directory and file name
        mask_path = os.path.join(mask_dir, name)
        mask_data = np.uint8(load_nifti(mask_path))
        
        # Apply preprocessing steps to the mask data:
        # Remove unwanted labels as specified by 'remove_label_ids'
        prepped_mask_data, z_indices = dataMaskPrep.preprocess_mask(mask_data)
        
        # Load and process the corresponding image data:
        # Load the image data from the specified directory and file name
        image_path = os.path.join(image_dir, name)
        image_data = load_nifti(image_path)
        
        prepped_image_data = dataImagePrep.preprocess_image(image_data, z_indices)

        # prepped_bbox_labels = dataAnnotationPrep.preprocess_box(prepped_mask_data)

        # Derive the base filename for saving processed files without their original extension
        base_name = name.split(file_suffix)[0]

        # import pdb; pdb.set_trace()

        # Save the processed image and mask data
        save_processed_data(prepped_image_data, prepped_mask_data, z_indices, output_dir, base_name)

def yolo_slice_standardization(config_name):
    """
    Parses config parameters to load standardized NIfTI images and masks, preprocess for SAM, and save images and masks
    as slices in .npy files.
    """
    cfg = load_dataset_config(config_name, root)

    # # Run the SAM preparation process with parameters from the config
    # dataMaskPrep = MaskPrep(
    #     remove_label_ids=cfg.get('yolo_preprocessing_cfg').get('remove_label_ids'),
    #     target_label_id=cfg.get('yolo_preprocessing_cfg').get('target_label_id', None),  # Optional parameter with default
    #     voxel_threshold_3d=cfg.get('yolo_preprocessing_cfg').get('voxel_num_thre3d'),
    #     pixel_threshold_2d=cfg.get('yolo_preprocessing_cfg').get('voxel_num_thre2d'),
    #     image_size_tuple=(cfg.get('yolo_preprocessing_cfg').get('image_size'),
    #                       cfg.get('yolo_preprocessing_cfg').get('image_size')),
    #     crop_non_zero_slices_flag = True,
    #     )

    # dataImagePrep = ImagePrep(
    #     image_size_tuple=(cfg.get('yolo_preprocessing_cfg').get('image_size'),
    #                       cfg.get('yolo_preprocessing_cfg').get('image_size')),
    # )

    # import pdb; pdb.set_trace()

    dataMaskPrep = MaskPrep(
        remove_label_ids=cfg.get('yolo_preprocessing_cfg').get('remove_label_ids'),
        target_label_id=cfg.get('yolo_preprocessing_cfg').get('target_label_id', None),  # Optional parameter with default
        voxel_threshold_3d=cfg.get('yolo_preprocessing_cfg').get('voxel_num_thre3d'),
        pixel_threshold_2d=cfg.get('yolo_preprocessing_cfg').get('voxel_num_thre2d'),
        image_size_tuple=cfg.get('yolo_preprocessing_cfg').get('image_size'),
        crop_non_zero_slices_flag = True,
        )
    
    dataImagePrep = ImagePrep(image_size_tuple=cfg.get('yolo_preprocessing_cfg').get('image_size'))

    nifti_to_npy(
        mask_dir=cfg.get('nifti_mask_dir'),
        image_dir=cfg.get('nifti_image_dir'),
        dataMaskPrep=dataMaskPrep,
        dataImagePrep=dataImagePrep,
        output_dir=cfg.get('yolo_npy_dir'),
        file_suffix=".nii.gz"  # Assuming this is a constant suffix for all files
    )
    return


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess medical data for SAM analysis.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()

    # Load the configuration file
    config_name = args.config_name + '.yaml'

    yolo_slice_standardization(config_name)