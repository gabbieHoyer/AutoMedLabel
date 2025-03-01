
import cv2
import cc3d
import numpy as np

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
# from src.preprocessing.dev.sam_prep import MaskPrep, ImagePrep
from src.preprocessing.model_prep import MaskPrep

# -------------------- DATA POST-PROCESS FUNCTIONS -------------------- #
def postprocess_resize(mask, image_size_tuple:tuple[int,int], make_square):
    """Resize mask to new dimensions."""
    predMaskPrep = MaskPrep(make_square=make_square)
    resized_mask = predMaskPrep.resize_mask(mask_data = mask.astype(np.uint8),
                                            mask_size_tuple = image_size_tuple)
    return resized_mask

def resize_prediction(sam_pred, image_size_tuple:tuple[int,int], label_id:int, make_square):
    """Convert SAM prediction into segmentation mask with original image dimensions and assign label_id."""
    sam_mask_resized = postprocess_resize(sam_pred, image_size_tuple, make_square)
    sam_mask = np.zeros_like(sam_mask_resized, dtype=np.uint8)
    sam_mask[sam_mask_resized > 0] = label_id
    return sam_mask

def refine_autolabel_mask(sam_pred):
    """
    Refine an autolabel mask.
    Used in the autolabel pipeline.
    Converts a binary mask to a refined mask via Gaussian blur and thresholding.
    """
    # Ensure sam_pred is in the range [0, 255] and of type np.uint8
    sam_pred = sam_pred.astype(np.uint8) * 255  # Convert binary 0/1 to grayscale 0/255
    
    # Apply Gaussian blur
    smooth_mask = cv2.GaussianBlur(sam_pred, (5, 5), 0)
    
    # Apply binary thresholding to convert back to binary mask
    _, processed_mask = cv2.threshold(smooth_mask, 127, 1, cv2.THRESH_BINARY)
    
    # Convert to np.uint8 if necessary
    processed_mask = processed_mask.astype(np.uint8)
    
    return processed_mask

# Function to remove small objects in 2D
def remove_small_objects_2D(mask_data, pixel_threshold_2d):
    """Remove small objects from 2D mask data using connected components."""
    mask_data = cc3d.dust(mask_data, threshold=pixel_threshold_2d, connectivity=8, in_place=True)
    return mask_data

# Post-processing function for predicted masks
def refine_evaluation_mask(sam_pred, pixel_threshold_2d, kernel_size=5):
    """
    Refine an evaluation mask.
    Used in the evaluation pipeline if the user opts for extra post‐processing.
    Applies connected components analysis to remove small objects,
    followed by morphological closing, Gaussian blur, and thresholding.
    """    
    # Ensure sam_pred is in the range [0, 255] and of type np.uint8
    sam_pred = sam_pred.astype(np.uint8) * 255  # Convert binary 0/1 to grayscale 0/255

    # Remove small objects using connected components analysis
    cleaned_mask = remove_small_objects_2D(sam_pred, pixel_threshold_2d)

    # Create the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological closing to the cleaned mask (dilation followed by erosion)
    closed_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Apply Gaussian blur to the morphologically closed mask
    smooth_mask = cv2.GaussianBlur(closed_mask, (kernel_size, kernel_size), 0)

    # Apply binary thresholding to convert back to binary mask
    _, processed_mask = cv2.threshold(smooth_mask, 127, 1, cv2.THRESH_BINARY)

    # Convert to np.uint8 if necessary
    processed_mask = processed_mask.astype(np.uint8)

    return processed_mask

def resize_boxes(boxes, current_size, target_size):
    """
    Resize bounding boxes from current size to target size.
    
    Parameters:
    - boxes: numpy array of shape (N, 4) where each box is [x_min, y_min, x_max, y_max]
    - current_size: tuple (width, height) of the current size of the image
    - target_size: tuple (width, height) of the target size of the image
    
    Returns:
    - Resized bounding boxes as a numpy array of shape (N, 4)
    """
    scale_x = target_size[0] / current_size[0]
    scale_y = target_size[1] / current_size[1]
    
    resized_boxes = boxes.copy()
    resized_boxes[:, 0] *= scale_x  # x_min
    resized_boxes[:, 1] *= scale_y  # y_min
    resized_boxes[:, 2] *= scale_x  # x_max
    resized_boxes[:, 3] *= scale_y  # y_max
    
    return resized_boxes