import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

# from src.preprocessing.dev.yolo_prep import ImagePrep
from src.preprocessing.model_prep import ImagePrep
from src.utils.file_management.file_handler import load_data
from src.utils.file_management.config_handler import summarize_config

from ultralytics import YOLO, RTDETR
# from infer_helper import *

from src.obj_detection.utils.visualization_utils import (
    visualize_full_pred, visualize_input, visualize_pred, set_image_clim
)
from src.obj_detection.utils.file_utils import (
    save_prediction, save_prediction_for_ITK, extract_filename, determine_run_directory, locate_files, load_dcm
)
from src.obj_detection.utils.model_utils import (
    make_predictor, make_sam2_predictor, get_model_pathway, make_prediction, resize_prediction, postprocess_prediction, load_config
)

def autoLabel(
    det_model,
    segmentation_model,
    segmentation_model_name,
    img_3D_unprocessed,
    dataImagePrep,
    img_name,
    mask_labels,
    remove_label_ids=[],
    conf=0.5,
    visualize=False,
    img_clim=False,
    run_dir=None,
    device='cpu'
):
    # Note image processing is in 2 steps. Step 2 is performed on a per slice basis
    img_3D = dataImagePrep.prep_image_step1(img_3D_unprocessed)
    if img_clim:
        image_clim = set_image_clim(img_3D)
    else: 
        image_clim=None
        
    mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)  # mask_3D_orig_size.shape -> (27, 640, 1024)
    orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])

    sam_model = segmentation_model.to(device)

    for slice_idx in range(img_3D.shape[0]):
        img_2D = dataImagePrep.prep_image_step2(img_3D[slice_idx,:,:])  
        img_2D_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)  # would happen in the sam finetuning dataset class  (1024, 1024, 3)
        # ----------- #
        # if visualize and slice_idx % 3 == 0:
        #     visualize_input(image=img_2D_3c, image_name=f"{img_name}_{slice_idx}", model_save_path=run_dir)
        # ----------- #
        det_results = det_model(source=img_2D_3c, conf=conf, device=device)  # conf=0.5

        # Convert the shape to (3, H, W)
        img_2D_3c = np.transpose(img_2D_3c, (2, 0, 1)) # would happen in the sam finetuning dataset class (3, 1024, 1024)
        img_2D_3c = torch.tensor(img_2D_3c).float()  # happens in sam dataset when batching
        img_2D_3c = img_2D_3c.to(device)

        combined_mask = torch.zeros((img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2]), dtype=torch.float32).to(device)

        for result in det_results:
            class_ids = result.boxes.cls.int().tolist()  # noqa
            if len(class_ids):
                boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            # ----------------------------------------------------- #    
            for label_id, bbox in zip(class_ids, boxes):
                if label_id in remove_label_ids:
                    continue

                bbox = bbox.to(device)

                # Call the appropriate segmentation model
                if segmentation_model_name == 'SAM':
                    sam_pred = segmentation_model(
                        img_2D_3c.unsqueeze(0),
                        bbox.unsqueeze(0)
                    )
                elif segmentation_model_name == 'SAM2':
                    sam_pred = segmentation_model(
                        img_2D_3c.unsqueeze(0),
                        bbox.unsqueeze(0)
                    )
                    # If SAM2 requires additional processing, add it here
                else:
                    raise ValueError(f"Unsupported segmentation model: {segmentation_model_name}")

                pred_binary = torch.sigmoid(sam_pred) > 0.5 # Convert logits to binary predictions

                # ----------- #
                # if visualize: 
                #     visualize_pred(image=img_2D_3c.detach().cpu(),
                #                 pred_mask=sam_pred.squeeze().detach().cpu(),
                #                 binary_pred=pred_binary.squeeze().detach().cpu(),
                #                 boxes=bbox.detach().cpu().numpy(), 
                #                 image_clim=image_clim,
                #                 image_name=f"{img_name}_{slice_idx}_labelID_{label_id}",
                #                 model_save_path=run_dir)
                # ----------- #

                sam_mask = resize_prediction(pred_binary.squeeze().detach().cpu().numpy(), orig_img_with_bbox_size, label_id, dataImagePrep.make_square)
                # ----------- 
                sam_mask = postprocess_prediction(sam_mask)
                # ----------- 
                sam_mask = torch.tensor(sam_mask).to(device)

                # Update combined_mask only where sam_mask indicates presence and no earlier label exists
                mask_indices = (sam_mask > 0) & (combined_mask == 0)
                combined_mask[mask_indices] = label_id

        # Convert combined_mask to the final integer mask
        mask_3D_orig_size[slice_idx] = combined_mask.cpu().numpy().astype(np.uint8)
        
        if visualize and slice_idx % 5 == 0:  
            visualize_full_pred(image=img_3D_unprocessed[slice_idx,...],
                            pred_mask=mask_3D_orig_size[slice_idx,...],
                            mask_labels=mask_labels,
                            image_name=f"{img_name}_{slice_idx}",
                            model_save_path=run_dir,
                            image_clim=image_clim)
            
    return mask_3D_orig_size


def setup_system(data_cfg, preprocessing_cfg, detection_cfg, segmentation_cfg, output_cfg, run_dir, device):
    dataImagePrep = ImagePrep(
        image_size_tuple=(
            preprocessing_cfg.get('image_size', 1024),
            preprocessing_cfg.get('image_size', 1024)
        ),
        make_square=preprocessing_cfg.get('make_square', False),
    )

    det_model = YOLO(os.path.join(root, detection_cfg.get('model_path')))

    # Get model_type from configuration
    model_type = segmentation_cfg.get('model_type', 'vit_b')

    # Determine which segmentation model to use based on model_type
    model_pathway = get_model_pathway(model_type)

    if model_pathway == 'SAM':
        segmentation_model = make_predictor(
            model_type=model_type,
            comp_cfg=segmentation_cfg.get('trainable', {}),
            initial_weights=os.path.join(root, segmentation_cfg.get('base_model')),
            finetuned_weights=os.path.join(root, segmentation_cfg.get('model_path')),
            device=device
        )
    elif model_pathway == 'SAM2':
        segmentation_model = make_sam2_predictor(
            comp_cfg=segmentation_cfg.get('trainable', {}),
            sam2_model_cfg=model_type,
            initial_weights=os.path.join(root, segmentation_cfg.get('base_model')),
            finetuned_weights=os.path.join(root, segmentation_cfg.get('model_path')),
            device=device
        )

    visualize_enabled = output_cfg['visualize']
    img_clim = output_cfg.get('img_clim', False)

    med_files = locate_files(data_cfg['data_dir'])
    for i, image_path in enumerate(tqdm(med_files)):
        if os.path.isdir(image_path):
            img_3D_unprocessed = load_dcm(image_path)
        else:
            img_3D_unprocessed = load_data(image_path)

        img_name = extract_filename(image_path)

        pred_volume = autoLabel(
            det_model,
            segmentation_model,
            model_pathway,
            img_3D_unprocessed,
            dataImagePrep,
            img_name,
            data_cfg['mask_labels'],
            preprocessing_cfg['remove_label_ids'],
            detection_cfg['conf'],
            visualize_enabled,
            img_clim,
            run_dir,
            device
        )

        save_prediction(pred_volume, run_dir, filename=img_name, output_ext=output_cfg['output_ext'])
        # save_prediction_for_ITK(pred_volume, run_dir, filename=img_name, output_ext=output_cfg['output_ext'])
        
        # # Check condition after the third iteration
        # if i == 2 and visualize_enabled:
        #     visualize_enabled = False  # Disable visualization from this point onwards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLO Model using Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    base_dir = os.getcwd()
    config_name = args.config_name + '.yaml'
    cfg = load_config(config_name, base_dir)

    run_dir = determine_run_directory(
        cfg.get('output_cfg').get('base_output_dir'),
        cfg.get('output_cfg').get('task_name')
    )

    summarize_config(cfg, path=os.path.join(root, run_dir, 'Run_Summaries'))

    setup_system(
        data_cfg=cfg.get('data'),
        preprocessing_cfg=cfg.get('preprocessing_cfg'),
        detection_cfg=cfg.get('models').get('obj_det'),
        segmentation_cfg=cfg.get('models').get('segmentation'),
        output_cfg=cfg.get('output_cfg'),
        run_dir=run_dir,
        device=cfg.get('device')
    )