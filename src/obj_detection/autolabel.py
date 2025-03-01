import os
import logging
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

from src.preprocessing import ImagePrep
from src.finetuning.engine import load_segmentation_model, load_detection_model
from src.utils import (
    set_image_clim,
    ALV,
    resize_prediction,
    refine_autolabel_mask,
    determine_run_directory,
    get_base_name,
    locate_files,
    load_dcm,
    load_data,
    save_prediction,
    save_prediction_for_ITK,
    summarize_config,
    load_autolabel_config,
)

logger = logging.getLogger(__name__)

def autoLabel_impl(
    det_model,
    segmentation_model,
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
        if visualize and slice_idx % 20 == 0:
            # visualize_input
            ALV.plot_raw_input_image(image=img_2D_3c, image_name=f"{img_name}_{slice_idx}", model_save_path=run_dir)
        # ----------- #
        det_results = det_model(source=img_2D_3c, conf=conf, device=device)  # conf=0.5
        if not det_results or not hasattr(det_results[0], 'boxes') or len(det_results[0].boxes) == 0:
            print(f"No detections for image slice {slice_idx}")
            continue
    
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

                sam_pred = sam_model(
                    img_2D_3c.unsqueeze(0),
                    bbox.unsqueeze(0)
                )
                pred_binary = torch.sigmoid(sam_pred) > 0.5 # Convert logits to binary predictions

                # ----------- #
                if visualize and slice_idx % 10 == 0: 
                    # visualize_pred
                    ALV.plot_prediction_and_binary_overlay(
                        image=img_2D_3c.detach().cpu(),
                        pred_mask=sam_pred.squeeze().detach().cpu(),
                        binary_pred=pred_binary.squeeze().detach().cpu(),
                        boxes=bbox.detach().cpu().numpy(), 
                        image_clim=image_clim,
                        image_name=f"{img_name}_{slice_idx}_labelID_{label_id}",
                        model_save_path=run_dir
                    )
                # ----------- #
                sam_mask = resize_prediction(pred_binary.squeeze().detach().cpu().numpy(), orig_img_with_bbox_size, label_id, dataImagePrep.make_square)
                # ----------- 
                sam_mask = refine_autolabel_mask(sam_mask)
                # ----------- 
                sam_mask = torch.tensor(sam_mask).to(device)

                # Update combined_mask only where sam_mask indicates presence and no earlier label exists
                mask_indices = (sam_mask > 0) & (combined_mask == 0)
                combined_mask[mask_indices] = label_id

        # Convert combined_mask to the final integer mask
        mask_3D_orig_size[slice_idx] = combined_mask.cpu().numpy().astype(np.uint8)
        
        if visualize and slice_idx % 10 == 0:  
            # visualize_full_pred
            ALV.plot_complete_prediction_overlay(
                image=img_3D_unprocessed[slice_idx,...],
                pred_mask=mask_3D_orig_size[slice_idx,...],
                mask_labels=mask_labels,
                image_name=f"{img_name}_{slice_idx}",
                model_save_path=run_dir,
                image_clim=image_clim
            )
    return mask_3D_orig_size


def system_setup(cfg):
    # Create the run directory and print a summary of the configuration.
    run_dir = determine_run_directory(cfg.get('output_cfg').get('base_output_dir'),
                                      cfg.get('output_cfg').get('task_name'))
    summarize_config(cfg, path=os.path.join(root, run_dir, 'Run_Summaries'))

    # Load and configure the models and preprocessing.
    data_cfg = cfg.get('data', {})
    prep_cfg = cfg.get('preprocessing_cfg', {})
    det_cfg = cfg.get('models').get('obj_det', {})
    seg_cfg = cfg.get('models').get('segmentation', {})
    output_cfg = cfg.get('output_cfg', {})
    device = cfg.get('device')

    dataImagePrep = ImagePrep(
        image_size_tuple=(prep_cfg.get('image_size', 1024), prep_cfg.get('image_size', 1024)),
        make_square=prep_cfg.get('make_square', False),
    )
    det_model = load_detection_model(det_cfg)
    trainable_cfg = seg_cfg.get('trainable', {})
    trainable_cfg.setdefault('resize_masks', True)

    model_config = seg_cfg.get('model_type', 'vit_b')
    weights_path = os.path.join(root, seg_cfg.get('pretrain_model'))
    checkpoint_path = os.path.join(root, seg_cfg.get('finetuned_model'))

    seg_model = load_segmentation_model(trainable_cfg, model_config, weights_path, device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            seg_model.load_state_dict(checkpoint["model"])
        except Exception as e:
            logger.error("Failed to load checkpoint from " + checkpoint_path + ": " + str(e))
            raise e

    visualize_enabled = output_cfg.get('visualize', False)
    img_clim = output_cfg.get('img_clim', False)
    med_files = locate_files(data_cfg['data_dir'])
    for i, image_path in enumerate(tqdm(med_files)):
        if os.path.isdir(image_path):
            img_3D_unprocessed = load_dcm(image_path)
        else:
            img_3D_unprocessed = load_data(image_path, use_sitk=prep_cfg.get('use_sitk', False))
        img_name = get_base_name(image_path)
        pred_volume = autoLabel_impl(
            det_model,
            seg_model,
            img_3D_unprocessed,
            dataImagePrep,
            img_name,
            data_cfg['mask_labels'],
            prep_cfg.get('remove_label_ids', []),
            det_cfg.get('conf', 0.5),
            visualize_enabled,
            img_clim,
            run_dir,
            device
        )
        save_prediction(pred_volume, run_dir, filename=img_name,
                        output_ext=output_cfg.get('output_ext'),
                        save_method=output_cfg.get('save_method', "nibabel"))

def autoLabel(config):
    """
    Public entry point. Accepts either a configuration filename (string) or a configuration dictionary.
    """
    if isinstance(config, str):
        base_dir = os.getcwd()
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        cfg = load_autolabel_config(config_file, base_dir)
    else:
        cfg = config
    system_setup(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLO Model using Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()
    autoLabel(args.config_name)

