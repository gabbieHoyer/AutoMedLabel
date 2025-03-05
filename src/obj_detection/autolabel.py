import os
import copy
import torch
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.preprocessing import ImagePrep
from src.finetuning.engine import load_segmentation_model, load_detection_model
from src.utils import (
    set_image_clim,
    set_image_norm,
    compute_image_clim_log, 
    compute_clamped_percentiles,
    ALV,
    resize_prediction,
    refine_autolabel_mask,
    determine_run_directory,
    in_notebook,
    get_base_name,
    locate_files,
    load_dcm,
    load_data,
    save_prediction,
    summarize_config,
    load_autolabel_config,
)

logger = logging.getLogger(__name__)

def default_agg_callback(combined_mask, confidence_map, new_mask, new_conf, label_id, mask_labels):
    # Instead of using confidence_map, update only where combined_mask is still 0.
    update_indices = (new_mask > 0) & (combined_mask == 0)
    combined_mask[update_indices] = label_id
    # Optionally, you can update the confidence_map as well if needed:
    confidence_map[update_indices] = new_conf
    return combined_mask, confidence_map

# ------------ Base Class Definition ------------ #

class AutoLabelBase:
    def __init__(self, config, callback=None, final_callback=None):
        self.config = config
        self.callback = callback
        self.final_callback = final_callback
        self.data_cfg = config.get('data', {})
        self.prep_cfg = config.get('preprocessing_cfg', {})
        self.det_cfg = config.get('models', {}).get('obj_det', {})
        self.seg_cfg = config.get('models', {}).get('segmentation', {})
        self.output_cfg = config.get('output_cfg', {})
        self.device = config.get('device', 'cpu')
        
        # Determine output directory and summarize config.
        self.run_dir = determine_run_directory(self.output_cfg.get('base_output_dir'),
                                                self.output_cfg.get('task_name'))
        summarize_config(config, path=os.path.join(self.run_dir, 'Run_Summaries'))
        
        # Initialize the image preprocessor.
        self.dataImagePrep = ImagePrep(
            image_size_tuple=(self.prep_cfg.get('image_size', 1024), self.prep_cfg.get('image_size', 1024)),
            make_square=self.prep_cfg.get('make_square', False)
        )
        
        # Load the object detection and segmentation models.
        self.det_model = load_detection_model(self.det_cfg)
        trainable_cfg = self.seg_cfg.get('trainable', {})
        trainable_cfg.setdefault('resize_masks', True)
        model_config = self.seg_cfg.get('model_type', 'vit_b')
        weights_path = os.path.join(os.getcwd(), self.seg_cfg.get('pretrain_model'))
        checkpoint_path = os.path.join(os.getcwd(), self.seg_cfg.get('finetuned_model'))
        self.seg_model = load_segmentation_model(trainable_cfg, model_config, weights_path, self.device)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.seg_model.load_state_dict(checkpoint["model"])
            except Exception as e:
                logger.error("Failed to load checkpoint from " + checkpoint_path + ": " + str(e))
                raise e

        self.visualize_enabled = self.output_cfg.get('visualize', False)
        self.img_clim = self.output_cfg.get('img_clim', False)
        self.desired_visuals = self.output_cfg.get('desired_visuals', 5)
        # Use the default aggregation callback; can be overridden.
        self.agg_callback = default_agg_callback

    def run_inference(self):
        med_files = locate_files(self.data_cfg['data_dir'])
        results = []
        for image_path in tqdm(med_files):
            # Load the image volume.
            if os.path.isdir(image_path):
                img_3D_unprocessed = load_dcm(image_path)
            else:
                img_3D_unprocessed = load_data(image_path, use_sitk=self.prep_cfg.get('use_sitk', False))
            img_name = get_base_name(image_path)
            pred_volume = self._process_case(img_3D_unprocessed, img_name)
            save_prediction(pred_volume, self.run_dir, filename=img_name,
                            output_ext=self.output_cfg.get('output_ext'),
                            save_method=self.output_cfg.get('save_method', "nibabel"))
            self.on_final_visualization(img_3D_unprocessed, pred_volume, img_name)
            results.append(pred_volume)
        return results

    def _process_case(self, img_3D_unprocessed, img_name):
        # Preprocess the volume.
        img_3D = self.dataImagePrep.prep_image_step1(img_3D_unprocessed)
        self.image_clim = set_image_norm(img_3D) if self.img_clim else None
        # self.image_clim = compute_clamped_percentiles(img_3D) if self.img_clim else None

        mask_3D_orig_size = np.zeros_like(img_3D_unprocessed, dtype=np.uint8)
        orig_img_with_bbox_size = (img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2])
        self.seg_model.to(self.device)
        n_slices = img_3D.shape[0]
        if self.desired_visuals == 1:
            visual_indices = [n_slices // 2]
        else:
            visual_interval = max(1, n_slices // self.desired_visuals)
            visual_indices = list(range(0, n_slices, visual_interval))
            if visual_indices[-1] != n_slices - 1:
                visual_indices.append(n_slices - 1)
        confidence_map = torch.zeros((img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2]),
                                     dtype=torch.float32).to(self.device)
        for slice_idx in range(n_slices):
            img_2D = self.dataImagePrep.prep_image_step2(img_3D[slice_idx, :, :])
            img_2D_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)
            det_results = self.det_model(source=img_2D_3c, conf=self.det_cfg.get('conf', 0.5),
                                         device=self.device, verbose=False)
            if not det_results or not hasattr(det_results[0], 'boxes') or len(det_results[0].boxes) == 0:
                continue

            # Convert image to (3, H, W)
            img_2D_3c = np.transpose(img_2D_3c, (2, 0, 1))
            img_2D_3c = torch.tensor(img_2D_3c).float().to(self.device)
            combined_mask = torch.zeros((img_3D_unprocessed.shape[1], img_3D_unprocessed.shape[2]),
                                        dtype=torch.float32).to(self.device)

            for result in det_results:
                # Extract detection items: label, bbox, and confidence.
                det_items = list(zip(result.boxes.cls.int().tolist(),
                                     result.boxes.xyxy,
                                     result.boxes.conf.tolist()))
                # Sort by priority defined in mask_labels.
                det_items_sorted = sorted(det_items, key=lambda x: self.data_cfg['mask_labels'].get(x[0], 9999))
                for label_id, bbox, detection_conf in det_items_sorted:
                    if label_id in self.prep_cfg.get('remove_label_ids', []):
                        continue
                    bbox = bbox.to(self.device)
                    sam_pred = self.seg_model(img_2D_3c.unsqueeze(0), bbox.unsqueeze(0))
                    pred_binary = torch.sigmoid(sam_pred) > 0.5
                    # Allow hook for intermediate visualization.
                    if self.visualize_enabled and slice_idx in visual_indices:
                        self.on_intermediate_visualization(
                            image=img_2D_3c.detach().cpu(),
                            pred_mask=sam_pred.squeeze().detach().cpu(),
                            binary_pred=pred_binary.squeeze().detach().cpu(),
                            boxes=bbox.detach().cpu().numpy(),
                            image_name=f"{img_name}_slice_{slice_idx}_labelID_{label_id}"
                        )
                    sam_mask_np = resize_prediction(pred_binary.squeeze().detach().cpu().numpy(),
                                                    orig_img_with_bbox_size,
                                                    label_id,
                                                    self.dataImagePrep.make_square)
                    sam_mask_np = refine_autolabel_mask(sam_mask_np)
                    sam_mask = torch.tensor(sam_mask_np).to(self.device)
                    combined_mask, confidence_map = self.agg_callback(
                        combined_mask, confidence_map, sam_mask, detection_conf, label_id, self.data_cfg['mask_labels']
                    )
            mask_3D_orig_size[slice_idx] = combined_mask.cpu().numpy().astype(np.uint8)
            if self.visualize_enabled and slice_idx in visual_indices:
                self.on_complete_visualization(
                    image=img_3D_unprocessed[slice_idx, ...],
                    pred_mask=mask_3D_orig_size[slice_idx, ...],
                    mask_labels=self.data_cfg['mask_labels'],
                    image_name=f"{img_name}_slice_{slice_idx}"
                )
        return mask_3D_orig_size

    # Hook methods (do nothing by default)
    def on_intermediate_visualization(self, image, pred_mask, binary_pred, boxes, image_name):
        pass

    def on_complete_visualization(self, image, pred_mask, mask_labels, image_name):
        pass

    def on_final_visualization(self, img_3D, pred_volume, img_name):
        pass

# ------------ Interactive Subclass ------------ #

class AutoLabelInteractive(AutoLabelBase):
    def __init__(self, config, callback=None, final_callback=None):
        if in_notebook():
            # If in a notebook, set default inline callback if none is provided.
            if callback is None:
                from IPython.display import display
                callback = lambda action, **kwargs: display(kwargs.get('fig'))
        else:
            # In terminal, set callback to None so that figures are saved.
            callback = None
        super().__init__(config, callback=callback, final_callback=final_callback)
    
    def on_intermediate_visualization(self, image, pred_mask, binary_pred, boxes, image_name):
        # Use existing visualization function via ALV.
        ALV.plot_prediction_and_binary_overlay(
            image=image,
            pred_mask=pred_mask,
            binary_pred=binary_pred,
            boxes=boxes,
            image_clim=None,  # add image_clim if needed.
            image_name=image_name,
            model_save_path=self.run_dir,
            callback=self.callback
        )

    def on_complete_visualization(self, image, pred_mask, mask_labels, image_name):
        ALV.plot_complete_prediction_overlay(
            image=image,
            pred_mask=pred_mask,
            mask_labels=mask_labels,
            image_name=image_name,
            model_save_path=self.run_dir,
            image_clim=self.image_clim,
            callback=self.callback
        )

    def on_final_visualization(self, img_3D, pred_volume, img_name):
        from IPython.display import display
        from src.utils.visualization.preprocessing.segmentation_plots import plot_segmentation_overlay

        # Define a helper to select slices (as in your notebook)
        def slice_num(data, desired_slices):
            z_index, _, _ = np.where(data > 0)
            possible_inds = np.unique(z_index)
            num_slices = len(possible_inds)
            step_ = max(1, int(np.ceil(num_slices / desired_slices)))
            start = (num_slices % desired_slices) // 2 if num_slices > desired_slices else 0
            return possible_inds[range(start, num_slices, step_)]

        slices = slice_num(pred_volume, desired_slices=24)
        # Generate the overview figure
        fig = plot_segmentation_overlay(
            vol=img_3D[slices, :, :],
            seg=pred_volume[slices, :, :],
            save_path=None,  # No saving, display inline
            seg_clim=self.img_clim,
            cmap='rainbow',
            title=f"{img_name} Prediction Overlay",
            labels_dict=self.data_cfg['mask_labels']
        )
        if fig is not None:
            if in_notebook():
                display(fig)
                plt.close(fig)
            else:
                # Not in a notebook, so save the figure instead.
                figure_file_path = os.path.join(self.run_dir, "final_QC", f"{img_name}_final.png")
                os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
                plt.savefig(figure_file_path, bbox_inches='tight', dpi=300)
                plt.close(fig)

# ------------------ Efficient Subclass ------------------ #

class AutoLabelEfficient(AutoLabelBase):
    def on_intermediate_visualization(self, image, pred_mask, binary_pred, boxes, image_name):
        # Do nothing in efficient mode.
        pass

    def on_complete_visualization(self, image, pred_mask, mask_labels, image_name):
        # No inline visualization.
        pass

    def on_final_visualization(self, img_3D, pred_volume, img_name):
        # Perhaps log minimal information instead of generating figures.
        print(f"Final prediction for {img_name} processed.")

# ------------ Run Function (Intermediate Step) ------------ #

def run_autolabel(config, interactive=False, callback=None, final_callback=None, agg_callback=None):
    # Allow config to be either a string (filename) or a dict.
    if isinstance(config, str):
        base_dir = os.getcwd()
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        cfg = load_autolabel_config(config_file, base_dir)
    else:
        cfg = config

    # Instantiate the proper pipeline based on interactive flag and environment.
    if interactive:
        if in_notebook():
            # In a notebook, use the provided callback if available,
            # otherwise default to an inline display callback.
            if callback is None:
                from IPython.display import display
                callback = lambda action, **kwargs: display(kwargs.get('fig'))
            pipeline = AutoLabelInteractive(cfg, callback=callback, final_callback=final_callback)
        else:
            # Running in a terminal: interactive mode but without an inline display callback,
            # so figures will be saved to disk.
            pipeline = AutoLabelInteractive(cfg, callback=None, final_callback=final_callback)
    else:
        pipeline = AutoLabelEfficient(cfg, callback=callback, final_callback=final_callback)
    
    if agg_callback is not None:
        pipeline.agg_callback = agg_callback

    pipeline.run_inference()

# ------------ Main __main__ Block for autolabel.py ------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict with YOLO Model using Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode (with visualizations)")
    args = parser.parse_args()

    run_autolabel(args.config_name, interactive=args.interactive)



