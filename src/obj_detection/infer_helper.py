import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path

from ultralytics import YOLO, RTDETR

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
from src.finetuning.engine.models.sam import finetunedSAM

import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import random



def map_labels_to_colors(pred_mask, mask_labels):
    # Define a colormap that can provide a distinct color for each class
    color_map = plt.get_cmap('tab20', len(mask_labels))  # 'tab20' has 20 distinct colors

    # Create an empty RGBA image
    colored_mask = np.zeros((*pred_mask.shape, 4), dtype=np.float32)  # Initialize with zeros

    # Map each label to a consistent color from the colormap
    for label_value, label_name in mask_labels.items():
        if label_value == 0:  # Skip the background
            continue
        mask = (pred_mask == label_value)
        color = color_map(label_value / len(mask_labels))  # Get consistent RGBA color
        colored_mask[mask] = color  # Apply color where the label matches

    return colored_mask


def visualize_pred(image, pred_mask, boxes, image_name, model_save_path):  # pred_mask_bin, label_id,
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with three subplots

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

    # Draw bounding boxes on the original image
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    figure_file_path = os.path.join(model_save_path, 'qual_check', f"{image_name}_pred_viz.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory


# --------------------------------------------------------------------------------

def visualize_pred_full(image, gt_mask, pred_mask, boxes, mask_labels, image_name, model_save_path):  # pred_mask_bin, label_id,
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

    # Original image
    # Assuming image is in [C, H, W] format and is an RGB image
    ax[0].imshow(image.permute(1, 2, 0))  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Ground truth mask
    ax[1].imshow(gt_mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    # Prediction overlay
    ax[2].imshow(image.permute(1, 2, 0))  # Original image
    colored_pred_mask = map_labels_to_colors(pred_mask.cpu().numpy(), mask_labels)
    ax[2].imshow(colored_pred_mask, alpha=0.5)  # Overlay the colored mask
    ax[2].set_title('Prediction Overlay')
    ax[2].axis('off')

    # Draw bounding boxes on the original image
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    figure_file_path = os.path.join(model_save_path, 'test_eval', f"{image_name}_pred_viz.png")
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

    figure_file_path = os.path.join(model_save_path, 'qual_check', f"{image_name}_yolo_qc.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path) 
    plt.close(fig)  # Close the figure to free memory

# -----------------------------------------------------------------

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


def get_all_files(directory_or_file):
    if os.path.isfile(directory_or_file):
        return [directory_or_file]
    elif os.path.isdir(directory_or_file):
        return [os.path.join(directory_or_file, f) for f in os.listdir(directory_or_file) if os.path.isfile(os.path.join(directory_or_file, f))]
    else:
        raise ValueError(f"{directory_or_file} is not a valid file or directory")

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

def make_predictor(model_type:str, initial_weights:str, finetuned_weights:str=None, device:str="cpu"):

    # Torch not compiled with cuda enabled :,D whaaaaattttt?
    # pointing to my lib/python3.11/site-packages/torch/nn/modules

    # import pdb; pdb.set_trace()

    sam_model = sam_model_registry[model_type](checkpoint=initial_weights).to(device)

    # import pdb; pdb.set_trace()

    # Check if a checkpoint exists to resume training
    if finetuned_weights and os.path.isfile(finetuned_weights):
        try:
            checkpoint = torch.load(finetuned_weights, map_location=device)

            # import pdb; pdb.set_trace()

            sam_model.load_state_dict(checkpoint["model"])

            # import pdb; pdb.set_trace()
            
        except Exception as e:
            # Decide whether to continue with training from scratch or to abort
            raise e
        
    # predictor = SamPredictor(sam_model)

    return sam_model


# def make_prediction(predictor, img_slice, bbox):
#     """Predict segmentation for image (2D) from bounding box prompt"""
#     img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)

#     predictor.set_image(img_3c.astype(np.uint8))

#     sam_mask, _, _ = predictor.predict(point_coords=None, 
#                                        point_labels=None, 
#                                        box=bbox[None, :], 
#                                        multimask_output=False)
#     return sam_mask

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








# def make_predictor(sam_model_type:str, sam_ckpt_path:str, device_:str=None):
#     """Return SAM predictor."""

#     import pdb; pdb.set_trace()

#     SAM_MODEL_TYPE = sam_model_type
#     SAM_CKPT_PATH = sam_ckpt_path

#     if not device_:
#         device_ = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device_)

#     sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
    
    
#     sam_model.to(device)
#     predictor = SamPredictor(sam_model)
#     return predictor

    # # Model finetuning setup
    # finetuned_model = finetunedSAM(
    #     image_encoder=sam_model.image_encoder,
    #     mask_decoder=sam_model.mask_decoder,
    #     prompt_encoder=sam_model.prompt_encoder,
    #     config=comp_cfg
    # ).to(device)

    # # Check if finetuned_weights are the same as initial_weights
    # if finetuned_weights == initial_weights:
    #     return finetuned_model

    # # Check if a checkpoint exists to resume training
    # if finetuned_weights and os.path.isfile(finetuned_weights):
    #     try:
    #         checkpoint = torch.load(finetuned_weights, map_location=device)
    #         start_epoch = checkpoint["epoch"] + 1
    #         finetuned_model.load_state_dict(checkpoint["model"])
    #     except Exception as e:
    #         # Decide whether to continue with training from scratch or to abort
    #         raise e

    # return finetuned_model