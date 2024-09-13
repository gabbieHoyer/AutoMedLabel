

# make_predictor
# make_sam2_predictor
# get_model_pathway
# make_prediction

import os
import cv2
import yaml
import torch
import numpy as np

from segment_anything import sam_model_registry
from ultralytics import YOLO, RTDETR

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.preprocessing.model_prep import MaskPrep

from src.sam2.build_sam import build_sam2
from src.finetuning.engine.models.sam import finetunedSAM
from src.finetuning.engine.models.sam2 import finetunedSAM2, finetunedSAM2_1024

# ----------------------------------------------------------- #

def get_model_pathway(model_type):
    if model_type in sam_model_registry:
        return 'SAM'
    elif model_type.endswith('.yaml'):
        return 'SAM2'
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


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


# ----------------------------- SAM2 Predictor ------------------------------------ #
def make_sam2_predictor(comp_cfg, sam2_model_cfg:str, initial_weights:str, finetuned_weights:str=None, device:str="cpu"):

    # sam_model = sam_model_registry[model_type](checkpoint=initial_weights).to(device)

    sam2_checkpoint = initial_weights
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    finetuned_model = finetunedSAM2_1024(
        model=sam2_model,
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

# --------------------------------------------------------------------------------- #

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

# ------------------------------------------------------- #

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