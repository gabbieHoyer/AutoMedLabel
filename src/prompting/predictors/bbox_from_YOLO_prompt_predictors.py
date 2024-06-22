import numpy as np
#from skimage import measure  # Import measure module from skimage for connected components

import torch
from segment_anything import sam_model_registry, SamPredictor

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


from src.preprocessing.sam_prep import MaskPrep #, ImagePrep
from src.utils.bbox import identify_bbox_from_volume, identify_bbox_from_slice, adjust_bbox_to_new_img_size, \
                            identify_instance_bbox_from_volume, identify_instance_bbox_from_slice


#TODO

# -------------------- MAIN PIPELINE FUNCTIONS -------------------- #
def infer_3D_stepwise_prep(predictor, img_3D_unprocessed:np.array, gt_3D_unprocessed:np.array, \
                           dataImagePrep, dataMaskPrep, bbox_shift:int=0, instance_bboxes:bool=False):
    """
    Applies 3D study specific sam prep functions to volume, then 2D required sam prep functions to 
    slice prior to prediction. Bounding box prompts are extracted from prepped 3D mask with original dims, 
    then reshaped prior to sam prediction.
    Params:
    - predictor: Model to predict segmentations
    - image: 3D image [slice x height x width] for inference.
    - mask: 3D mask [slice x height x width] to extract bounding box.
    - dataImagePrep: Class to preprocess images for SAM.
    - dataMaskPrep: Class to preprocess masks for SAM.
    - bbox_shift: (Optional) Jitter for bbox (pixels).
    Returns:
    - Prediction with labels in the same dimensions as input mask.
    - Preprocessed ground truth in the same dimensions as input images.
    """
    ## Predict Segmentations
    
    #Preprocess data
    #Note: If not cropping, z_indices = [] and default functionality will include all slices
    #Note cont: If crop and no non_zero will raise a warning
    gt_3D, z_indices = dataMaskPrep.prep_mask_study_specific_step1(gt_3D_unprocessed)
    #Note image processing is in 2 steps. Step 2 is performed on a per slice basis
    img_3D = dataImagePrep.prep_image_study_specific_step1(img_3D_unprocessed, z_indices)

    # Extract bbox
    if instance_bboxes is True:
        volume_bbox_list = identify_instance_bbox_from_volume(gt_3D, bbox_shift)
    else:
        volume_bbox_list = identify_bbox_from_volume(gt_3D, bbox_shift)
    # If 'gts' is None, handle it appropriately
    if not any(volume_bbox_list):
        raise ValueError(f"No bounding box prompts found.")
    
    #Predict segmentations
    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])
    for slice_idx, slice_bbox_list in volume_bbox_list:
        img_2D = dataImagePrep.prep_image_sam_specific_step2(img_3D[slice_idx,:,:])
        new_img_with_bbox_size = (img_2D.shape[0], img_2D.shape[1])

        # Process each slice and get the bounding boxes
        for label_id, bbox in slice_bbox_list:
            adjusted_bbox = adjust_bbox_to_new_img_size(bbox, orig_img_with_bbox_size, new_img_with_bbox_size) 
            sam_pred = make_prediction(predictor, img_2D, adjusted_bbox)
            sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)
            mask_3D_orig_size[slice_idx, sam_mask > 0] = label_id 
    
    gt_3D_prep_orig_size = postprocess_resize(gt_3D, (gt_3D_unprocessed.shape[1], gt_3D_unprocessed.shape[2]))

    return mask_3D_orig_size, gt_3D_prep_orig_size

def infer_3D(predictor, img_3D_unprocessed:np.array, gt_3D_unprocessed:np.array, \
             dataImagePrep=None, dataMaskPrep=None, bbox_shift:int=0, instance_bboxes:bool=False):
    """
    Applies all sam prep steps at once. This includes 3D study specific sam prep functions to volume and 2D 
    required sam prep functions to slices. Bounding box prompts are extracted from prepped 3D mask with sam dims. 
    Output sam prediction is reshaped to original input image size.
    Params:
    - predictor: Model to predict segmentations
    - image: 3D image [slice x height x width] for inference.
    - mask: 3D mask [slice x height x width] to extract bounding box.
    - dataImagePrep: (Optional) Class to preprocess images for SAM.
    - dataMaskPrep: (Optional) Class to preprocess masks for SAM.
    - bbox_shift: (Optional) Jitter for bbox (pixels).
    Returns:
    - Prediction with labels in the same dimensions as input mask.
    - Preprocessed ground truth in the same dimensions as input images.
    """
    
    #Preprocess data
    #Note: If not cropping, z_indices = [] and default functionality will include all slices
    #Note cont: If crop and no non_zero will raise a warning
    if (dataMaskPrep) and (dataImagePrep):
        gt_3D, z_indices = dataMaskPrep.preprocess_mask(gt_3D_unprocessed)
        img_3D = dataImagePrep.preprocess_image(img_3D_unprocessed, z_indices)
    elif (dataMaskPrep) or (dataImagePrep):
        raise ValueError("Insufficient information to preprocess image and mask")
    else:
        gt_3D = gt_3D_unprocessed
        img_3D = img_3D_unprocessed
    
    # Extract bbox
    if instance_bboxes is True:
        volume_bbox_list = identify_instance_bbox_from_volume(gt_3D, bbox_shift)
    else:
        volume_bbox_list = identify_bbox_from_volume(gt_3D, bbox_shift)
    # If 'gts' is None, handle it appropriately
    if not any(volume_bbox_list):
        raise ValueError(f"No bounding box prompts found.")

    #Predict segmentations
    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])
    for slice, slice_bbox_list in volume_bbox_list:
        img_2D = img_3D[slice,:,:]
        
        # Process each slice and get the bounding boxes
        for label_id, bbox in slice_bbox_list:
            sam_pred = make_prediction(predictor, img_2D, bbox)
            sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)
            mask_3D_orig_size[slice, sam_mask > 0] = label_id 
    
    gt_3D_prep_orig_size = postprocess_resize(gt_3D, (gt_3D_unprocessed.shape[1], gt_3D_unprocessed.shape[2]))
    
    return mask_3D_orig_size, gt_3D_prep_orig_size