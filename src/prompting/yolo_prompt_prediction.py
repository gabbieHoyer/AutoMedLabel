import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path

from ultralytics import SAM, YOLO

import os
import random
import numpy as np
from tqdm import tqdm

import argparse


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

# -------------------- MAIN PIPELINE FUNCTIONS -------------------- #
def infer_3D_stepwise_prep(det_model, predictor, img_3D_unprocessed:np.array, gt_3D_unprocessed:np.array, \
                           dataImagePrep, device='cpu'):

    #Note image processing is in 2 steps. Step 2 is performed on a per slice basis
    img_3D = dataImagePrep.prep_image_study_specific_step1(img_3D_unprocessed)
    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])

    for slice_idx in img_3D.shape[0]:  # confirm accuracy

        img_2D = dataImagePrep.prep_image_sam_specific_step2(img_3D[slice_idx,:,:])
        
        # ----------------------------------------------------- #
        # add yolo inference per numpy array slice 
        det_results = det_model(img_2D, stream=True, device=device)

        for result in det_results:
            class_ids = result.boxes.cls.int().tolist()  # noqa
            if len(class_ids):
                boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        # ----------------------------------------------------- #
                for label_id, bbox in zip(class_ids, boxes):
                    sam_pred = make_prediction(predictor, img_2D, bbox)
                    sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)
                    mask_3D_orig_size[slice_idx, sam_mask > 0] = label_id 
    
    return mask_3D_orig_size


# -------------------- MAIN PIPELINE FUNCTIONS -------------------- #
def predict_segmentation(infer_method, det_model, predictor, selected_files, dataImagePrep, inference_output_path, device):
    # Predict Segmentations
    if infer_method == '3D_stepwise':
        for image_path in tqdm(selected_files):
            img_3D_unprocessed = load_data(image_path)
            mask_orig_size = infer_3D_stepwise_prep(det_model, predictor, img_3D_unprocessed, dataImagePrep, device)
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))
            save_prediction(mask_orig_size, inference_output_path, file_name_no_ext)


def prompt_prediction(cfg):
    # -------------------- DEFINE PARAMS -------------------- #   
    seg_model_path = os.path.join(root, cfg.get('models').get('segmentation').get('model_path'))
    det_model_path = os.path.join(root, cfg.get('models').get('obj_det').get('model_path'))

    predictor = make_predictor(sam_model_type = cfg.get('models').get('segmentation').get('model_type'), 
                        sam_ckpt_path = seg_model_path, 
                        device_ = cfg.get('device', 'cuda:0'),
                        )

    det_model = YOLO(det_model_path)
    
    # Select data to infer
    selected_files = cfg.get('nifti_data')

    if not (selected_files):
        raise ValueError("No input files identified.")

    dataImagePrep = ImagePrep(
        image_size_tuple=(cfg.get('preprocessing_cfg').get('image_size', 1024),
                        cfg.get('preprocessing_cfg').get('image_size', 1024)),  
        )

    infer_method = cfg.get('preprocessing_cfg').get('infer_method')

    inference_output_path = os.path.join(cfg.get('base_output_dir'), cfg.get('inference_dir'))

    predict_segmentation(infer_method, det_model, predictor, selected_files, dataImagePrep, inference_output_path, device= cfg.get('device', 'cuda:0'))

    return

if __name__ == "__main__":
    try: 
        parser = argparse.ArgumentParser(description="Infer MRI volume on SAM and variants.")
        parser.add_argument("config_name", help="Name of the YAML configuration file")
        parser.add_argument('--prompt', type=str, default='zeroshot', required=False, choices=['zeroshot', 'finetuned'], help='Select prompt: zeroshot for inference of dataset on baseline model weights (prompting run from dataset yaml config), finetuned for inference of dataset(s) on finetuned model weights (prompting run from finetuning experiment yaml config)')
        args = parser.parse_args()
        config_name = args.config_name + '.yaml'   # Get the config file name from the command line argument

        cfg = load_prompting_experiment(config_name, root, args.prompt)
        import pdb; pdb.set_trace()

        prompt_prediction(cfg)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # path tbd but proves functionality
        summarize_config(cfg, path=cfg.get('output_configuration').get('save_path'))


