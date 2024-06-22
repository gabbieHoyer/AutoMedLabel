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

# -------------------- MODEL FUNCTIONS -------------------- #
def make_predictor(sam_model_type:str, sam_ckpt_path:str, device_:str=None):
    """Return SAM predictor."""
    SAM_MODEL_TYPE = sam_model_type
    SAM_CKPT_PATH = sam_ckpt_path

    if not device_:
        device_ = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_)
    sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    return predictor

def make_prediction(predictor, img_slice, bbox):
    """Predict segmentation for image (2D) from bounding box prompt"""
    img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)
    predictor.set_image(img_3c.astype(np.uint8))
    sam_mask, _, _ = predictor.predict(point_coords=None, 
                                       point_labels=None, 
                                       box=bbox[None, :], 
                                       multimask_output=False)
    return sam_mask


# -------------------- DATA POST-PROCESS FUNCTIONS -------------------- #
def postprocess_resize(mask, image_size_tuple:tuple[int,int]):
    """Resize mask to new dimensions."""
    predMaskPrep = MaskPrep()
    resized_mask = predMaskPrep.resize_mask(mask_data = mask.astype(np.uint8),
                                            image_size_tuple = image_size_tuple)
    return resized_mask

def postprocess_prediction(sam_pred, image_size_tuple:tuple[int,int], label_id:int):
    """Convert SAM prediction into segmentation mask with original image dims and label ids."""
    sam_mask_resized = postprocess_resize(sam_pred, image_size_tuple)
    sam_mask = np.zeros_like(sam_mask_resized, dtype=np.uint8)
    sam_mask[sam_mask_resized > 0] = label_id
    return sam_mask

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
    gt_3D_resized = dataMaskPrep.prep_mask_sam_specific_step2(gt_3D)
    if instance_bboxes is True:
        volume_bbox_list = identify_instance_bbox_from_volume(gt_3D_resized, bbox_shift)
    else:
        volume_bbox_list = identify_bbox_from_volume(gt_3D_resized, bbox_shift)
    # If 'gts' is None, handle it appropriately
    if not any(volume_bbox_list):
        raise ValueError(f"No bounding box prompts found.")
    
    #Predict segmentations
    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])
    for slice_idx, slice_bbox_list in volume_bbox_list:
        img_2D = dataImagePrep.prep_image_sam_specific_step2(img_3D[slice_idx,:,:])
        
        # Process each slice and get the bounding boxes
        for label_id, bbox in slice_bbox_list:
            #adjusted_bbox = adjust_bbox_to_new_img_size(bbox, orig_img_with_bbox_size, new_img_with_bbox_size) 
            sam_pred = make_prediction(predictor, img_2D, bbox)
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

def infer_2D(predictor, img_2D_unprocessed:np.array, gt_2D_unprocessed:np.array, \
             dataImagePrep=None, dataMaskPrep=None, bbox_shift:int=0, instance_bboxes:bool=False):
    """
    All preprocessing steps are applied to 2D image. Output sam prediction is reshaped to original input image size.
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
        gt_2D, z_indices = dataMaskPrep.preprocess_mask(gt_2D_unprocessed)
        img_2D = dataImagePrep.preprocess_image(img_2D_unprocessed, z_indices)
    elif (dataMaskPrep) or (dataImagePrep):
        raise ValueError("Insufficient information to preprocess image and mask")
    else:
        gt_2D = gt_2D_unprocessed
        img_2D = img_2D_unprocessed

    # Extract bbox
    if instance_bboxes is True:
        slice_bbox_list = identify_instance_bbox_from_slice(gt_2D, bbox_shift)
    else:
        slice_bbox_list = identify_bbox_from_slice(gt_2D, bbox_shift)
    # If 'gts' is None, handle it appropriately
    if not any(slice_bbox_list):
        raise ValueError(f"No bounding box prompts found.")
    
    #Predict segmentations
    mask_2D_orig_size = np.zeros_like(img_2D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_2D_unprocessed.shape[0], img_2D_unprocessed.shape[1])
    # Process each slice and get the bounding boxes
    for label_id, bbox in slice_bbox_list:
        sam_pred = make_prediction(predictor, img_2D, bbox)
        sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)
        mask_2D_orig_size[ sam_mask > 0] = label_id 

    gt_2D_prep_orig_size = postprocess_resize(gt_2D, (gt_2D_unprocessed.shape[0], gt_2D_unprocessed.shape[1]))

    return mask_2D_orig_size, gt_2D_prep_orig_size


# -------------------- MAIN PIPELINE DEBUGGING FUNCTIONS -------------------- #
def infer_3D_debug(predictor, img_3D_unprocessed:np.array, gt_3D_unprocessed:np.array, \
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
    # --- for debug ---
    mask_3D_model_size_debug = np.zeros_like(img_3D, dtype=np.uint8)
    # --- end debug ---
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])
    for slice, slice_bbox_list in volume_bbox_list:
        img_2D = img_3D[slice,:,:]
        
        # Process each slice and get the bounding boxes
        for label_id, bbox in slice_bbox_list:
            sam_pred = make_prediction(predictor, img_2D, bbox)
            sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)
            mask_3D_orig_size[slice, sam_mask > 0] = label_id 
            # --- for debug ---
            sam_mask_debug = sam_pred[0].astype(np.uint8)
            mask_3D_model_size_debug[slice, sam_mask_debug > 0] = label_id 
            # --- end debug ---
    
    gt_3D_prep_orig_size = postprocess_resize(gt_3D, (gt_3D_unprocessed.shape[1], gt_3D_unprocessed.shape[2]))
    # --- for debug ---
    return mask_3D_orig_size, gt_3D_prep_orig_size, img_3D, mask_3D_model_size_debug
    # --- end debug ---
    #return mask_3D_orig_size, gt_3D_prep_orig_size