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
import argparse
import os
import yaml
from ultralytics import YOLO
from ultralytics import RTDETR
from infer_helper import *


def infer_3D_stepwise_prep(det_model, predictor, img_3D_unprocessed:np.array, gt_3D_unprocessed:np.array, \
                           dataImagePrep, device='cpu'):

    #Note image processing is in 2 steps. Step 2 is performed on a per slice basis
    import pdb; pdb.set_trace()

    img_3D = dataImagePrep.prep_image_study_specific_step1(img_3D_unprocessed)

    import pdb; pdb.set_trace()

    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])

    import pdb; pdb.set_trace()
    for slice_idx in img_3D.shape[0]:  # confirm accuracy

        import pdb; pdb.set_trace()

        img_2D = dataImagePrep.prep_image_sam_specific_step2(img_3D[slice_idx,:,:])

        import pdb; pdb.set_trace()
        
        # ----------------------------------------------------- #
        # add yolo inference per numpy array slice 
        det_results = det_model(img_2D, stream=True, device=device)

        import pdb; pdb.set_trace()

        for result in det_results:
            class_ids = result.boxes.cls.int().tolist()  # noqa

            import pdb; pdb.set_trace()

            if len(class_ids):
                boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        # ----------------------------------------------------- #
                import pdb; pdb.set_trace()
                
                for label_id, bbox in zip(class_ids, boxes):

                    sam_pred = make_prediction(predictor, img_2D, bbox)

                    import pdb; pdb.set_trace()

                    sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)

                    import pdb; pdb.set_trace()

                    mask_3D_orig_size[slice_idx, sam_mask > 0] = label_id 
    
    import pdb; pdb.set_trace()
    
    return mask_3D_orig_size


def predict_segmentation(infer_method, det_model, predictor, selected_files, dataImagePrep, inference_output_path, device):
    # Predict Segmentations
    if infer_method == '3D_stepwise':
        for image_path in tqdm(selected_files):
            img_3D_unprocessed = load_data(image_path)
            mask_orig_size = infer_3D_stepwise_prep(det_model, predictor, img_3D_unprocessed, dataImagePrep, device)
            
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))

            import pdb; pdb.set_trace()
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
    
    selected_files = cfg.get('nifti_data')

    dataImagePrep = ImagePrep(
        image_size_tuple=(cfg.get('preprocessing_cfg').get('image_size', 1024),
                        cfg.get('preprocessing_cfg').get('image_size', 1024)),  
        )

    infer_method = cfg.get('preprocessing_cfg').get('infer_method')
    inference_output_path = os.path.join(cfg.get('base_output_dir'), cfg.get('inference_dir'))

    import pdb; pdb.set_trace()

    predict_segmentation(infer_method, det_model, predictor, selected_files, dataImagePrep, inference_output_path, device= cfg.get('device', 'cuda:0'))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLO Model using Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    base_dir = os.getcwd()  # Assumes the script is run from the project root
    config_name = args.config_name + '.yaml'
    config = load_config(config_name, base_dir)

    # predict(config)
    prompt_prediction(config)



# def predict(config):

#     ModelClass = load_model(config)
#     model = ModelClass(config['yolo_weights'])

#     # results = model(source=config['data'], conf=config.get('conf', 0.5), save=True, save_txt=True)
#     results = model(source=config['nifti_data'], conf=config.get('conf', 0.5), save=True, save_txt=True)
    
#     print(f"Prediction results saved: {results}")






