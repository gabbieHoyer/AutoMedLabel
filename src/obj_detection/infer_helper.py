import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path

from ultralytics import YOLO

import os
import random
import numpy as np
from tqdm import tqdm

import argparse
import argparse
import os
import yaml


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


from src.preprocessing.sam_prep import MaskPrep, ImagePrep
from src.utils.file_management.config_handler import load_prompting_experiment, summarize_config, update_cfg_for_dataset
from src.utils.file_management.file_handler import load_json
from src.utils.file_management.path_info import pair_files_in_split, pair_files, file_without_extension_from_path

from src.utils.file_management.file_handler import load_data


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

# -------------------- DATA SAVING FUNCTIONS -------------------- #
def save_prediction(seg_3D, save_dir, file_name_no_ext):
    """Save prediction as .npz file with keys 'seg' and 'gt'."""
    # Ensure the updated paths exist
    os.makedirs(save_dir, exist_ok=True)

    # Assuming file_name is already the base filename without the extension
    npz_output_path = os.path.join(save_dir, f"{file_name_no_ext}.npz")
    np.savez_compressed(npz_output_path, seg=seg_3D) 
    return

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