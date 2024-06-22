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

from src.utils.file_management.config_handler import load_prompting_experiment, summarize_config, update_cfg_for_dataset
from src.prompting.predictors.bbox_from_mask_prompt_predictors import make_predictor, infer_3D_stepwise_prep, infer_3D, infer_2D, infer_3D_debug
from src.preprocessing.sam_prep import MaskPrep, ImagePrep

from src.utils.file_management.file_handler import load_json
from src.utils.file_management.path_info import pair_files_in_split, pair_files, file_without_extension_from_path

# -------------------- DATA PREP FUNCTIONS -------------------- #
from src.utils.file_management.file_handler import load_data

def select_files(image_dir:str, mask_dir:str, path_ext:str='', num_files:int=None, splits_json_file_path:dict={}, split_text:dict={}):
    """Identify paired (image, mask files). Only identify paired files in split is specified."""
    # Load the JSON file if specified   
    if splits_json_file_path and split_text:
        split_dict = load_json(splits_json_file_path)
        paired_files = pair_files_in_split(image_dir, mask_dir, path_ext, split_dict, split_text)
    else:
        paired_files = pair_files(image_dir, mask_dir, path_ext)
    
    # Randomly select files if only selecting a few pairs
    if num_files:
        selected_pairs = random.sample(paired_files, min(len(paired_files), num_files))
    else:
        selected_pairs = paired_files 
    return selected_pairs

# -------------------- DATA SAVING FUNCTIONS -------------------- #
def save_prediction(seg_3D, gts, save_dir, file_name_no_ext):
    """Save prediction as .npz file with keys 'seg' and 'gt'."""
    # Ensure the updated paths exist
    os.makedirs(save_dir, exist_ok=True)

    # Assuming file_name is already the base filename without the extension
    npz_output_path = os.path.join(save_dir, f"{file_name_no_ext}.npz")
    np.savez_compressed(npz_output_path, seg=seg_3D, gt=gts) 
    #print(f"Inference results saved to {npz_output_path}")
    return

# -------------------- MAIN PIPELINE FUNCTIONS -------------------- #
def predict_segmentation(infer_method, predictor, selected_files, dataImagePrep, dataMaskPrep, bbox_shift, inference_output_path):
    # Predict Segmentations
    if infer_method == '3D_stepwise':
        for image_path, mask_path in tqdm(selected_files):
            img_3D_unprocessed = load_data(image_path)
            gt_3D_unprocessed = load_data(mask_path)  # Assuming masks are in .nii.gz format
            mask_orig_size, gt_prep_orig_size = infer_3D_stepwise_prep(predictor, img_3D_unprocessed, gt_3D_unprocessed, dataImagePrep, dataMaskPrep, bbox_shift)
            # Derive the base filename for saving processed files without their original extension
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))
            save_prediction(mask_orig_size, gt_prep_orig_size, inference_output_path, file_name_no_ext)

    elif infer_method == '3D':
        for image_path, mask_path in tqdm(selected_files):
            mask_orig_size, gt_prep_orig_size = infer_3D(predictor, img_3D_unprocessed, gt_3D_unprocessed, dataImagePrep, dataMaskPrep, bbox_shift)
            # Derive the base filename for saving processed files without their original extension
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))
            save_prediction(mask_orig_size, gt_prep_orig_size, inference_output_path, file_name_no_ext)

    elif infer_method == '2D':
        for image_path, mask_path in tqdm(selected_files):
            mask_orig_size, gt_prep_orig_size = infer_2D(predictor, img_3D_unprocessed, gt_3D_unprocessed, dataImagePrep, dataMaskPrep, bbox_shift)
            # Derive the base filename for saving processed files without their original extension
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))
            save_prediction(mask_orig_size, gt_prep_orig_size, inference_output_path, file_name_no_ext)

    # --- for debug ---
    elif infer_method == '3D_debug':
        from src.utils.visualizers import plot_segmentation_overlay
        for image_path, mask_path in tqdm(selected_files):
            img_3D_unprocessed = load_data(image_path)
            gt_3D_unprocessed = load_data(mask_path) 
            mask_orig_size, gt_prep_orig_size, img_3D, mask_3D_model_size_debug = infer_3D_debug(predictor, img_3D_unprocessed, gt_3D_unprocessed, dataImagePrep, dataMaskPrep, bbox_shift, instance_bboxes_flag)
            # Derive the base filename for saving processed files without their original extension
            file_name_no_ext = file_without_extension_from_path(os.path.basename(image_path))
            save_prediction(mask_orig_size, gt_prep_orig_size, inference_output_path, file_name_no_ext)

            # Assuming file_name is already the base filename without the extension
            save_dir = inference_output_path.replace('preds','figs_debug_no_postproc')
            # Ensure the updated paths exist
            os.makedirs(save_dir, exist_ok=True)
            debug_plot_savepath = os.path.join(save_dir, f"{file_name_no_ext}_debug.png")
            
            # find non-zero slices
            z_index, _, _ = np.where(mask_3D_model_size_debug > 0)
            possible_inds = np.unique(z_index)
            num_slices = len(possible_inds)
            step_ = max(1, int(np.ceil(num_slices / 18)))
            start = (num_slices % 18) // 2 if num_slices > 18 else 0
            slices = possible_inds[range(start, num_slices, step_)]
            
            plot_segmentation_overlay(vol= img_3D[slices,:,:], 
                                    seg= mask_3D_model_size_debug[slices,:,:], 
                                    seg_clim = [np.min(mask_3D_model_size_debug), np.max(mask_3D_model_size_debug)],
                                    save_path=debug_plot_savepath, 
                                    cmap='rainbow', 
                                    title=file_name_no_ext, 
                                    ) 
    # --- end debug ---

def prompt_prediction(cfg):
    """
    - inference_output_path: Output directory to save images.
    Outputs:
    - Save ground truth and prediction in inference_output_path/file_name.npz with keys 'seg' and 'gt'.
    """
    # -------------------- DEFINE PARAMS -------------------- #
    for model_cfg in cfg['models']:
        dynamic_cfg_list = update_cfg_for_dataset(cfg, model_cfg)
        for dynamic_cfg in dynamic_cfg_list:

            model_path = os.path.join(root, model_cfg.get('model_path'))
            
            # NOTE: incorporate logic from gpu setup into zero shot scripts so not manual device setting
            predictor = make_predictor(sam_model_type = model_cfg.get('model_type'), 
                                    sam_ckpt_path = model_path, 
                                    device_ = cfg.get('device', 'cuda:0'),
                                    )
            
            # Select data to infer
            selected_files = select_files(image_dir = dynamic_cfg.get('input_cfg').get('image_path'), 
                                        mask_dir = dynamic_cfg.get('input_cfg').get('gt_path'), 
                                        path_ext = cfg.get('input_cfg').get('path_ext'), 
                                        splits_json_file_path = dynamic_cfg.get('input_cfg').get('ml_metadata_file',''), 
                                        split_text = cfg.get('input_cfg').get('split','')
                                        )
            if not (select_files):
                raise ValueError("No input files identified.")

            # Run the SAM preparation process with parameters from the config
            dataMaskPrep = MaskPrep(
                remove_label_ids=dynamic_cfg.get('preprocessing_cfg').get('remove_label_ids'),
                target_label_id=dynamic_cfg.get('preprocessing_cfg').get('target_label_id', None),  # Optional parameter with default
                voxel_threshold_3d=dynamic_cfg.get('preprocessing_cfg').get('voxel_num_thre3d'),
                pixel_threshold_2d=dynamic_cfg.get('preprocessing_cfg').get('voxel_num_thre2d'),
                image_size_tuple=(dynamic_cfg.get('preprocessing_cfg').get('image_size', 1024),
                                dynamic_cfg.get('preprocessing_cfg').get('image_size', 1024)),  # Default to 1024 if not specified
                crop_non_zero_slices_flag = cfg.get('preprocessing_cfg').get('crop_non_zero_slices_flag',None),
                )

            dataImagePrep = ImagePrep(
                image_size_tuple=(dynamic_cfg.get('preprocessing_cfg').get('image_size', 1024),
                                dynamic_cfg.get('preprocessing_cfg').get('image_size', 1024)),  # Default to 1024 if not specified
                )

            infer_method = cfg.get('preprocessing_cfg').get('infer_method')
            bbox_shift = cfg.get('preprocessing_cfg').get('bbox_shift')

            inference_output_path = os.path.join(dynamic_cfg.get('base_output_dir'), dynamic_cfg.get('inference_dir'))

            predict_segmentation(infer_method, predictor, selected_files, dataImagePrep, dataMaskPrep, bbox_shift, inference_output_path)

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













# def prompt_prediction(cfg):
#     """
#     - inference_output_path: Output directory to save images.
#     Outputs:
#     - Save ground truth and prediction in inference_output_path/file_name.npz with keys 'seg' and 'gt'.
#     """
#     # -------------------- DEFINE PARAMS -------------------- #
#     for model_cfg in cfg.get('models'):
#         #dynamic_cfg_list = update_cfg_for_dataset(cfg, model_cfg)
        
#         for dataset_name, dataset_cfg in cfg['datasets'].items():
        
#             model_path = os.path.join(root, model_cfg.get('model_path'))
            
#             # NOTE: incorporate logic from gpu setup into zero shot scripts so not manual device setting
#             predictor = make_predictor(sam_model_type = model_cfg.get('model_type'), 
#                                     sam_ckpt_path = model_path, 
#                                     device_ = cfg.get('device', 'cuda:0'),
#                                     )
            
#             # Select data to infer
#             selected_files = select_files(image_dir = dataset_cfg.get('input_cfg').get('image_path'), 
#                                         mask_dir = dataset_cfg.get('input_cfg').get('gt_path'), 
#                                         path_ext = cfg.get('input_cfg').get('path_ext'), 
#                                         splits_json_file_path = dataset_cfg.get('input_cfg').get('ml_metadata_file',''), 
#                                         split_text = cfg.get('input_cfg').get('split',''),
#                                         num_files=5,
#                                         )
#             if not (select_files):
#                 raise ValueError("No input files identified.")

#             # Run the SAM preparation process with parameters from the config
#             dataMaskPrep = MaskPrep(
#                 remove_label_ids = dataset_cfg.get('preprocessing_cfg').get('remove_label_ids', None),
#                 target_label_id = dataset_cfg.get('preprocessing_cfg').get('target_label_id', None),  # Optional parameter with default
#                 voxel_threshold_3d = dataset_cfg.get('preprocessing_cfg').get('voxel_num_thre3d'),
#                 pixel_threshold_2d = dataset_cfg.get('preprocessing_cfg').get('voxel_num_thre2d'),
#                 image_size_tuple = (dataset_cfg.get('preprocessing_cfg').get('image_size', 1024),
#                                 dataset_cfg.get('preprocessing_cfg').get('image_size', 1024)),  # Default to 1024 if not specified
#                 crop_non_zero_slices_flag = cfg.get('preprocessing_cfg').get('crop_non_zero_slices_flag',None),
#                 )

#             dataImagePrep = ImagePrep(
#                 image_size_tuple=(dataset_cfg.get('preprocessing_cfg').get('image_size', 1024),
#                                 dataset_cfg.get('preprocessing_cfg').get('image_size', 1024)),  # Default to 1024 if not specified
#                 )

#             infer_method = cfg.get('preprocessing_cfg').get('infer_method')
#             bbox_shift = cfg.get('preprocessing_cfg').get('bbox_shift')
#             instance_bboxes_flag = dataset_cfg.get('preprocessing_cfg').get('instance_bboxes_flag', False)
            
#             import pdb; pdb.set_trace()
#             output_path = cfg['base_output_dir'].replace('${project_output_dir}',dataset_cfg.get('project_output_dir'))\
#                             .replace('${experiment.name}',cfg.get('experiment').get('name'))\
#                             .replace("DATASET.name", dataset_name)\
#                             .replace('${models.model_weights}',model_cfg.get('model_weights'))
            
#             output_path = os.path.join(output_path, cfg.get('inference_dir'))
            
#             print(output_path)
#             predict_segmentation(infer_method, predictor, selected_files, dataImagePrep, dataMaskPrep, bbox_shift, instance_bboxes_flag, output_path)

#     return