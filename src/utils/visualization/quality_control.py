
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import random

# Assume these helper functions exist in your misc module.
from .helpers import upscale_mask, map_labels_to_colors, map_labels_to_contours_qc, add_colorbar_h

#############################
# Quality Control Visualizer
#############################
class QualityControlVisualizer:
    """
    Visualizer for quality control in the finetuning/evaluation pipelines.
    
    Methods:
      - plot_input_with_bbox_and_masks: Displays the raw input image with the box prompt overlaid,
        the ground‐truth mask, and the model’s raw prediction.
        (Formerly: visualize_input)
        
      - plot_segmentation_overlay: Displays the original image, the ground‐truth mask, and an overlay
        of the prediction with transparent color mapping and drawn bounding boxes.
        (Formerly: visualize_predictions)
        
      - plot_qc_input_with_bboxes: Generates a full‐scale quality check visualization from a batch.
        (Formerly: full_scale_visualize_input)
        
      - plot_prediction_contour_overlay: Displays an image with contours for both the ground‐truth and
        predicted masks (and optionally bounding boxes) for detailed QC.
        (Formerly: visualize_contour_predictions3_qc)
        
      - run_batch_quality_check: Randomly selects a batch from the loader and runs QC visualizations.
        (Formerly: quality_check)
    """

    @staticmethod
    def plot_input_with_bbox_and_masks(image, gt_mask, box, pred_mask, label_id, image_name, model_save_path):
        """Formerly: visualize_input"""
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # Original image
        ax[0].imshow(image.permute(1, 2, 0), cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        # Ground truth mask
        ax[1].imshow(gt_mask, cmap='rainbow')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        # Draw bounding box on the original image
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        # Predicted mask
        ax[2].imshow(pred_mask, cmap='rainbow')
        ax[2].set_title(f'Predicted Mask: {label_id}')
        ax[2].axis('off')
        figure_file_path = os.path.join(model_save_path, 'qual_check', f"{image_name}_input_viz.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

# -----
    @staticmethod
    def plot_gt_pred_mask_overlay(image, gt_mask, pred_mask, mask_labels, image_name, model_save_path):
        """Formerly: visualize_full_pred"""
        target_hw = (image.shape[1], image.shape[2])
        if gt_mask.shape[-2:] != target_hw:
            gt_mask = upscale_mask(gt_mask, target_hw)
        if pred_mask.shape[-2:] != target_hw:
            pred_mask = upscale_mask(pred_mask, target_hw)

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(image.permute(1, 2, 0), cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(image.permute(1, 2, 0), cmap='gray')
        colored_gt_mask = map_labels_to_colors(gt_mask, mask_labels)
        ax[1].imshow(colored_gt_mask, alpha=0.5)
        ax[1].set_title('Ground Truth Overlay')
        ax[1].axis('off')

        ax[2].imshow(image.permute(1, 2, 0), cmap='gray')
        colored_pred_mask = map_labels_to_colors(pred_mask, mask_labels)
        ax[2].imshow(colored_pred_mask, alpha=0.5)
        ax[2].set_title('Prediction Overlay')
        ax[2].axis('off')
        # Add colorbar (assume add_colorbar is defined elsewhere)
        # e.g., add_colorbar(fig, ax, mask_labels)
        figure_file_path = os.path.join(model_save_path, "combined_mask_QC", f"{image_name}.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)
# -----

    @staticmethod
    def plot_segmentation_overlay(image, gt_mask, pred_mask, boxes, mask_labels, image_name, model_save_path):
        """Formerly: visualize_predictions"""
        target_hw = (image.shape[1], image.shape[2])
        if gt_mask.shape[-2:] != target_hw:
            gt_mask = upscale_mask(gt_mask, target_hw)
        if pred_mask.shape[-2:] != target_hw:
            pred_mask = upscale_mask(pred_mask, target_hw)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # Original image
        ax[0].imshow(image.permute(1, 2, 0))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        # Ground truth mask
        ax[1].imshow(gt_mask, cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        # Prediction overlay
        ax[2].imshow(image.permute(1, 2, 0))
        colored_pred_mask = map_labels_to_colors(pred_mask.cpu().numpy(), mask_labels)
        ax[2].imshow(colored_pred_mask, alpha=0.5)
        ax[2].set_title('Prediction Overlay')
        ax[2].axis('off')
        # Draw bounding boxes on the original image
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
        figure_file_path = os.path.join(model_save_path, 'test_eval', f"{image_name}_pred_viz.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

    @staticmethod
    def plot_qc_input_with_bboxes(output_dir, image, gt_mask, boxes, label_id, image_name):
        """Formerly: full_scale_visualize_input"""
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        ax[0].set_title(f'Original Image with Boxes for Label ID: {label_id}')
        ax[0].axis('off')
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
        ax[1].imshow(gt_mask.cpu().numpy(), cmap='gray')
        ax[1].set_title(f'Ground Truth Mask for Label ID: {label_id}')
        ax[1].axis('off')
        figure_file_path = os.path.join(output_dir, 'qual_check', f"QC_visualization_{image_name}_label_{label_id}.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

    @staticmethod
    def plot_prediction_contour_overlay(image, gt_mask, pred_mask, boxes, mask_labels, image_name, model_save_path):
        """Formerly: visualize_contour_predictions3_qc"""
        target_hw = (image.shape[1], image.shape[2])
        if gt_mask.shape[-2:] != target_hw:
            gt_mask = upscale_mask(gt_mask, target_hw)
        if pred_mask.shape[-2:] != target_hw:
            pred_mask = upscale_mask(pred_mask, target_hw)
        fig, ax = plt.subplots(1, 5, figsize=(22, 6))
        fig.suptitle(f"{image_name}", fontsize=16)
        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.85, bottom=0.1, left=0.05, right=0.95)
        ax[0].imshow(image.permute(1, 2, 0))
        ax[0].set_title('Original Image + Box Prompts')
        ax[0].axis('off')
        ax[1].imshow(image.permute(1, 2, 0), cmap='gray')
        colored_gt_mask = map_labels_to_colors(gt_mask.cpu().numpy(), mask_labels)
        ax[1].imshow(colored_gt_mask, alpha=0.5)
        ax[1].set_title('Ground Truth Overlay')
        ax[1].axis('off')
        ax[2].imshow(image.permute(1, 2, 0), cmap='gray')
        colored_pred_mask = map_labels_to_colors(pred_mask.cpu().numpy(), mask_labels)
        ax[2].imshow(colored_pred_mask, alpha=0.5)
        ax[2].set_title('Prediction Overlay')
        ax[2].axis('off')
        ax[3].imshow(image.permute(1, 2, 0))
        ax[3].set_title('GT Contours')
        ax[3].axis('off')
        ax[4].imshow(image.permute(1, 2, 0))
        pred_contours_dict = map_labels_to_contours_qc(pred_mask.cpu().numpy(), mask_labels)
        gt_contours_dict = map_labels_to_contours_qc(gt_mask.cpu().numpy(), mask_labels)
        for label_value, contour_info in gt_contours_dict.items():
            color = contour_info['color']
            for contour in contour_info['contours']:
                if contour.shape[0] > 0:
                    ax[3].plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, label=f'GT: {label_value}')
        for label_value, contour_info in pred_contours_dict.items():
            color = contour_info['color']
            for contour in contour_info['contours']:
                if contour.shape[0] > 0:
                    ax[4].plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, label=f'Pred: {label_value}')
        ax[4].set_title('Prediction Contours')
        ax[4].axis('off')
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
        add_colorbar_h(fig, ax, mask_labels)
        figure_file_path = os.path.join(model_save_path, 'test_eval', f"{image_name}_pred_viz_contours.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

    @staticmethod
    def run_batch_quality_check(test_loader, output_dir):
        """Formerly: quality_check"""
        num_batches = len(test_loader)
        random_batch_idx = random.randint(0, num_batches - 1)
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx != random_batch_idx:
                continue
            img_names = batch['img_name']
            images, gt2D, boxes, label_ids = batch['image'], batch['gt2D'], batch['boxes'], batch['label_ids']
            for i in range(images.size(0)):
                image = images[i]
                img_name = img_names[i]
                label_boxes_masks = {}
                for j in range(len(label_ids[i])):
                    lab_id = label_ids[i][j].item()
                    if lab_id not in label_boxes_masks:
                        label_boxes_masks[lab_id] = {'boxes': [], 'mask': (gt2D[i, 0] == lab_id).float()}
                    label_boxes_masks[lab_id]['boxes'].append(boxes[i][j])
                for lab_id, data in label_boxes_masks.items():
                    QualityControlVisualizer.plot_qc_input_with_bboxes(output_dir, image, data['mask'], data['boxes'], lab_id, img_name)


#############################
# Autolabel Visualizer
#############################
class AutolabelVisualizer:
    """
    Visualizer for the autolabel pipeline.
    
    Methods:
      - plot_complete_prediction_overlay: Shows full prediction overlay with a colorbar.
        (Formerly: visualize_full_pred)
        
      - plot_raw_input_image: Displays only the raw input image.
        (Formerly: visualize_input, the version that only displays the image)
        
      - plot_prediction_and_binary_overlay: Shows both the raw prediction overlay and the binary prediction overlay with bounding boxes.
        (Formerly: visualize_pred)
    """

    @staticmethod
    def plot_complete_prediction_overlay(image, pred_mask, mask_labels, image_name, model_save_path, image_clim=None):
        """Formerly: visualize_full_pred"""
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray', clim=image_clim)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        if image_clim is not None:
            ax[1].imshow(image, cmap='gray', clim=image_clim)
        else:
            ax[1].imshow(image, cmap='gray')
        colored_pred_mask = map_labels_to_colors(pred_mask, mask_labels)
        ax[1].imshow(colored_pred_mask, alpha=0.5)
        ax[1].set_title('Prediction Overlay')
        ax[1].axis('off')
        # Add colorbar (assume add_colorbar is defined elsewhere)
        # e.g., add_colorbar(fig, ax, mask_labels)
        figure_file_path = os.path.join(model_save_path, "QC", f"{image_name}_full_pred.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

    @staticmethod
    def plot_raw_input_image(image, image_name, model_save_path):
        """Formerly: visualize_input (raw version)"""
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        figure_file_path = os.path.join(model_save_path, "input_QC", f"{image_name}.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

    @staticmethod
    def plot_prediction_and_binary_overlay(image, pred_mask, binary_pred, boxes, image_name, model_save_path, image_clim=None):
        """Formerly: visualize_pred"""
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray', clim=image_clim)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(image, cmap='gray', clim=image_clim)
        ax[1].imshow(pred_mask, alpha=0.5)
        ax[1].set_title('Prediction Overlay')
        ax[1].axis('off')
        ax[2].imshow(image, cmap='gray', clim=image_clim)
        ax[2].imshow(binary_pred, alpha=0.5)
        ax[2].set_title('Binary Prediction Overlay')
        ax[2].axis('off')
        if not isinstance(boxes, list):
            boxes = [boxes]
        for box in boxes:
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
        figure_file_path = os.path.join(model_save_path, "label_QC", f"{image_name}_pred.png")
        os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)
        plt.savefig(figure_file_path)
        plt.close(fig)

