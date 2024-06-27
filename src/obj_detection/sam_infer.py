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

from src.utils.bbox import identify_bbox_from_volume, identify_bbox_from_slice, adjust_bbox_to_new_img_size, \
                            identify_instance_bbox_from_volume, identify_instance_bbox_from_slice


from src.preprocessing.yolo_prep import MaskPrep, ImagePrep
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

def infer_3D_stepwise_prep(det_model, sam_model, img_3D_unprocessed:np.array, dataImagePrep, device='cpu'):

    #Note image processing is in 2 steps. Step 2 is performed on a per slice basis
    img_3D = dataImagePrep.prep_image_study_specific_step1(img_3D_unprocessed)

    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])

    for slice_idx in range(img_3D.shape[0]):

        img_2D = dataImagePrep.prep_image_sam_specific_step2(img_3D[slice_idx,:,:])  # img_3D[slice_idx,:,:] for thigh is (256, 512)
        # img_2D.shape -> (1024, 1024)
        # img_2D.dtype -> dtype('float64')

        img_2D_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)
        # img_2D_3c.dtype -> dtype('float64')

        # import pdb; pdb.set_trace()

        # img_2D_3c = np.uint8(img_2D_3c)

        out_dir = '/data/VirtualAging/users/ghoyer/correcting_rad_workflow/det2seg/AutoMedLabel/standardized_data/thigh_inference/QC/initial_load/'
        image_name = f"thigh_slice_{slice_idx}"

        visualize_input(img_2D_3c, image_name, out_dir)
        # import pdb; pdb.set_trace()

        # img_2D_3c.shape -> (1024, 1024, 3)
        # img_2D_3c.dtype -> dtype('float64')

        # needed for YOLO prediction function - Prepares input image before inference.
        # im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        
        # ----------------------------------------------------- #
        # add yolo inference per numpy array slice 
        # det_results = det_model(img_2D_3c, stream=True, device=device)

        det_results = det_model(source=img_2D_3c, conf=0.5, device=device)

        # det_results = det_model(source=img_2D_3c, conf=0.5, device=device)
        # what I do in predict_msk and it works fine
        # model(source=config['data'], conf=config.get('conf', 0.5), save=True, save_txt=True)

        # import pdb; pdb.set_trace()

        # Convert the shape to (3, H, W)
        # img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        img_2D_3c = np.transpose(img_2D_3c, (2, 0, 1)) # (3, 1024, 1024)

        for result in det_results:

            # boxes is empty :,D
            class_ids = result.boxes.cls.int().tolist()  # noqa

            # import pdb; pdb.set_trace()

            if len(class_ids):
                boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        # ----------------------------------------------------- #
                # import pdb; pdb.set_trace()
                
                for label_id, bbox in zip(class_ids, boxes):

                    import pdb; pdb.set_trace()

                    # sam_pred = make_prediction(predictor, img_2D_3c, bbox)
                            # prediction = self.model(images[img_idx].unsqueeze(0), box.unsqueeze(0))   

                    # numpy.ndarray has no attribute 'to'

                    img_2D_3c_2 = torch.tensor(img_2D_3c).float()
                    img_2D_3c_3 = img_2D_3c_2.to(device)
                    # bbox_2 = bbox.to(device) # this is a tensor already

                    # thing = sam_model(img_2D_3c_2, bbox)

                    # thing2 = sam_model(img_2D_3c_3, bbox)
                    # .unsqueeze(0), box.unsqueeze(0)
                    # thing3 = sam_model2(img_2D_3c_3.unsqueeze(0), bbox.unsqueeze(0))
                    sam_model2 = sam_model.to(device)

                    sam_pred = sam_model(img_2D_3c, bbox)

                    out_dir2 = '/data/VirtualAging/users/ghoyer/correcting_rad_workflow/det2seg/AutoMedLabel/standardized_data/thigh_inference/QC/initial_load/'
                    image_name = f"thigh_slice_{slice_idx}_labelID_{label_id}"
                    visualize_pred(img_2D_3c, image_name, out_dir2)

                    import pdb; pdb.set_trace()

                    sam_mask = postprocess_prediction(sam_pred[0], orig_img_with_bbox_size, label_id)

                    import pdb; pdb.set_trace()

                    mask_3D_orig_size[slice_idx, sam_mask > 0] = label_id 
    
    import pdb; pdb.set_trace()
    
    return mask_3D_orig_size


def predict_segmentation(infer_method, det_model, sam_model, input_files, dataImagePrep, inference_output_path, device):
    # Predict Segmentations
    if infer_method == '3D_stepwise':

        for image_path in tqdm(input_files):  # unsupported file format: s
            img_3D_unprocessed = load_data(image_path)

            # thigh volume is (15, 256, 512)
            mask_orig_size = infer_3D_stepwise_prep(det_model, sam_model, img_3D_unprocessed, dataImagePrep, device)
                                                  
            import pdb; pdb.set_trace()
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))

            import pdb; pdb.set_trace()
            save_prediction(mask_orig_size, inference_output_path, file_name_no_ext)


def prompt_prediction(cfg):
    # -------------------- DEFINE PARAMS -------------------- #   
    seg_model_path = os.path.join(root, cfg.get('models').get('segmentation').get('model_path'))
    det_model_path = os.path.join(root, cfg.get('models').get('obj_det').get('model_path'))
    
    sam_model = make_predictor(model_type=cfg.get('models').get('segmentation').get('model_type'), 
                               initial_weights=cfg.get('base_model'), 
                               finetuned_weights=seg_model_path, 
                               device=cfg.get('device', 'cuda:0')
                               )
    
    det_model = YOLO(det_model_path)
    
    input_files = get_all_files(cfg.get('nifti_data'))
        
    dataImagePrep = ImagePrep(
        image_size_tuple=(cfg.get('preprocessing_cfg').get('image_size', 1024),
                        cfg.get('preprocessing_cfg').get('image_size', 1024)),  
        )

    infer_method = cfg.get('preprocessing_cfg').get('infer_method')
    inference_output_path = os.path.join(cfg.get('base_output_dir'), cfg.get('inference_dir'))

    predict_segmentation(infer_method, det_model, sam_model, input_files, dataImagePrep, inference_output_path, device= cfg.get('device', 'cuda:0'))

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






