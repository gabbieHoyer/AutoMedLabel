import os
import logging
import pandas as pd
import torch
from monai.transforms import AsDiscrete

from .base_engine import BaseTester
from .metrics.metric_factory import load_metrics
from .metrics.metric_utils import load_meta, parse_image_name, extract_meta, extract_slice_number
from src.finetuning.utils.logging import main_process_only, log_info, wandb_log
from src.utils import refine_evaluation_mask, QCV

logger = logging.getLogger(__name__)

class StandardTester(BaseTester):
    def __init__(self, model, test_loader, eval_cfg, module_cfg,
                 datamodule_cfg, experiment_cfg, run_path,
                 device='cpu', data_type='full', visualize=False):
        super().__init__(model, test_loader, eval_cfg, module_cfg,
                         datamodule_cfg, experiment_cfg, run_path,
                         device, data_type, visualize)
        # No additional initializations are required here unless needed

    def evaluate_multilabel_test_set(self):

        # Optionally run quality check if visualization is enabled
        if self.module_cfg['visualize']:
            QCV.run_batch_quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        self.dice_metric.reset()
        self.IoU_metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                img_name = batch['img_name']
                images = batch['image'].to(self.device)
                gt2D = batch['gt2D'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                present_labels = torch.unique(gt2D[gt2D != 0]).tolist()
                batch_size, height, width = gt2D.size(0), gt2D.size(2), gt2D.size(3)
                combined_mask = torch.zeros((batch_size, height, width),
                                            device=self.device, dtype=torch.int64)

                for img_idx in range(batch_size):
                    for class_label in present_labels:
                        class_mask = torch.zeros((height, width), device=self.device)
                        class_boxes = boxes[img_idx][label_ids[img_idx] == class_label]
                        
                        for instance_idx, box in enumerate(class_boxes):
                            if box.sum() == 0:
                                continue

                            prediction = self.model(
                                images[img_idx].unsqueeze(0),
                                box.unsqueeze(0)
                            )

                            if self.module_cfg['visualize'] and batch_idx < 3 and self.data_type=='sampled':
                                gt_mask_single_label = (gt2D == class_label).float() * gt2D[0]
                                # visualize_input(
                                QCV.plot_input_with_bbox_and_masks(
                                    image=images[img_idx].detach().cpu(),
                                    gt_mask=gt_mask_single_label.detach().cpu().squeeze(),
                                    box=box.detach().cpu(),
                                    pred_mask=prediction.detach().cpu().squeeze(),
                                    label_id=class_label,
                                    image_name=f"{img_name[img_idx]}_class_label_{class_label}_instance{instance_idx}",
                                    model_save_path=self.model_save_path
                                )

                            prediction_binary = torch.sigmoid(prediction) > 0.5
                            class_mask += prediction_binary.squeeze().float()

                        combined_mask[img_idx][class_mask > 0] = class_label

                if self.visualize and batch_idx % 10 == 0 and self.data_type=='sampled':
                    # visualize_predictions(
                    QCV.plot_segmentation_overlay(
                        image=images[0].cpu(),
                        gt_mask=gt2D[0].squeeze(0).cpu(),
                        pred_mask=combined_mask.squeeze(0).cpu(),
                        boxes=boxes[0].cpu(),
                        mask_labels=self.datamodule_cfg['mask_labels'],
                        image_name=f"{self.data_type}_{batch['img_name'][0]}_{batch_idx}",
                        model_save_path=self.model_save_path
                    )

                # Use base class helper to compute metrics
                dice_score, IoU_score = self.compute_batch_metrics(combined_mask, gt2D)
                if self.data_type == "full":
                    self.update_slice_metrics(dice_score, IoU_score, img_name)

            self.aggregate_and_log_global_metrics()

    def evaluate_non_multilabel_test_set(self):

        if self.module_cfg['visualize'] and self.data_type == 'sampled':
            QCV.run_batch_quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                img_name = batch['img_name']
                images = batch['image'].to(self.device)
                gt2D = batch['gt2D'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                batch_size, height, width = gt2D.size(0), gt2D.size(2), gt2D.size(3)
                num_classes = self.datamodule_cfg['num_classes']
                multi_class_probs = torch.zeros((batch_size, num_classes, height, width), device=self.device)

                # Process the first set of boxes and labels
                for box, label_id in zip(boxes[0], label_ids[0]):
                    prediction = self.model(images, box.unsqueeze(0))
                    prediction_binary = torch.sigmoid(prediction) > 0.5
                    class_index = label_id
                    multi_class_probs[:, class_index, :, :] = torch.max(
                        multi_class_probs[:, class_index, :, :],
                        prediction_binary.float()
                    )

                multi_masks = multi_class_probs.argmax(dim=1)
                dice_score, IoU_score = self.compute_batch_metrics(multi_masks, gt2D)
                if self.data_type == "full":
                    self.update_slice_metrics(dice_score, IoU_score, img_name)

                if self.visualize and batch_idx < 2 and self.data_type == 'sampled':
                    QCV.plot_segmentation_overlay(
                        image=images[0].cpu(),
                        gt_mask=gt2D[0].squeeze(0).cpu(),
                        pred_mask=multi_masks.squeeze(0).cpu(),
                        boxes=boxes[0].cpu(),
                        mask_labels=self.datamodule_cfg['mask_labels'],
                        image_name=f"{batch['img_name'][0]}_{batch_idx}",
                        model_save_path=self.model_save_path
                    )

            self.aggregate_and_log_global_metrics()

# ---------------------------------------------------------------------- #
class Det2SegTester(BaseTester):
    def __init__(self, model, det_model, test_loader, eval_cfg, module_cfg,
                 det_model_cfg, datamodule_cfg, experiment_cfg, run_path,
                 device='cpu', data_type='full', visualize=False):
        super().__init__(model, test_loader, eval_cfg, module_cfg,
                         datamodule_cfg, experiment_cfg, run_path, device,
                         data_type, visualize)
        self.det_model = det_model.to(device)
        self.det_model_cfg = det_model_cfg
        self.pixel_threshold_2d = self.datamodule_cfg['voxel_num_thre2d']
        self.kernel_size = self.datamodule_cfg['kernel_size']
        self.combined_mask_order = self.datamodule_cfg.get('combined_mask_order', None)
        self.remove_label_ids = self.datamodule_cfg.get('remove_label_ids', None)
        self.resize_masks = self.eval_cfg['trainable'].get('resize_masks', False)
        self.postprocess_mask = self.det_model_cfg.get('post_processing')

    def evaluate_multilabel_test_set(self):

        # Use quality check if visualization is enabled
        if self.module_cfg['visualize']:
            QCV.run_batch_quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        self.dice_metric.reset()
        self.IoU_metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Get an image identifier and extract a slice number if available
                img_name = batch['img_name'][0]
                slice_number = extract_slice_number(img_name)

                images = batch['image'].to(self.device)
                gt2D = batch['gt2D'].to(self.device)

                batch_size, height, width = gt2D.size(0), gt2D.size(2), gt2D.size(3)
                combined_mask = torch.zeros((batch_size, height, width),
                                            device=self.device, dtype=torch.int64)

                # Process each image in the batch individually
                for img_idx in range(batch_size):
                    # Convert image tensor to numpy array with shape (H, W, C)
                    image_np = images[img_idx].cpu().numpy().transpose(1, 2, 0)
                    
                    det_results = self.det_model(source=image_np,
                                                 conf=self.det_model_cfg['conf'],
                                                 device=self.device)
                   
                    if not det_results or not hasattr(det_results[0], 'boxes') or len(det_results[0].boxes) == 0:
                        print(f"No detections for image {batch['img_name'][img_idx]}")
                        continue

                    # Get detection class IDs and bounding boxes
                    class_ids = det_results[0].boxes.cls.int().tolist()
                    boxes = det_results[0].boxes.xyxy
                    det_items = list(zip(class_ids, boxes))

                    if self.combined_mask_order:
                        det_items = sorted(det_items, key=lambda x: self.combined_mask_order.index(x[0]))

                    # Process each detection for the image
                    for label_id, bbox in det_items:
                        if label_id in self.remove_label_ids:
                            continue
                        bbox = bbox.to(self.device)
                        # Run segmentation model with the given bounding box
                        prediction = self.model(images[img_idx].unsqueeze(0), bbox.unsqueeze(0))
                        prediction_binary = torch.sigmoid(prediction) > 0.5

                        if self.postprocess_mask:
                            processed = refine_evaluation_mask(
                                prediction_binary.squeeze().float().detach().cpu().numpy(),
                                self.pixel_threshold_2d,
                                self.kernel_size
                            )
                            prediction_binary = torch.tensor(processed).to(self.device)
                        else:
                            prediction_binary = prediction_binary.squeeze().float()

                        # Combine prediction into the output mask:
                        # Only update pixels that are positive and not already assigned
                        mask_indices = (prediction_binary > 0) & (combined_mask[img_idx] == 0)
                        combined_mask[img_idx][mask_indices] = label_id

                # Optionally visualize a sample if conditions are met
                if self.visualize and slice_number is not None and slice_number % 10 == 0 and self.data_type == 'sampled':
                    QCV.plot_prediction_contour_overlay(
                        image=images[0].detach().cpu(),
                        gt_mask=gt2D[0].squeeze(0).detach().cpu(),
                        pred_mask=combined_mask.squeeze(0).detach().cpu(),
                        boxes=boxes.cpu().numpy() if 'boxes' in locals() else None,
                        mask_labels=self.datamodule_cfg['mask_labels'],
                        image_name=f"{self.data_type}_{img_name}_{batch_idx}",
                        model_save_path=self.model_save_path
                    )

                # Compute metrics using the helper from the base class
                dice_score, IoU_score = self.compute_batch_metrics(combined_mask, gt2D)
                if self.data_type == "full":
                    self.update_slice_metrics(dice_score, IoU_score, [img_name])

            self.aggregate_and_log_global_metrics()

    def test(self):
        """
        Override the base-class test method to skip the `if self.instance_bbox`
        check. We always run `evaluate_multilabel_test_set()` for Det2Seg.
        """
        log_info(f"Testing (Det2Seg) with data_type='{self.data_type}'...")

        # Always call evaluate_multilabel_test_set
        self.evaluate_multilabel_test_set()

        # Same post-processing steps as base-class
        self.save_test_results()

        if self.data_type == "full":
            self.save_metric_scores_to_csv(metric_name='dice', slice_metric=self.slice_dice_metrics)
            self.save_metric_scores_to_csv(metric_name='IoU', slice_metric=self.slice_IoU_metrics)

# ----------------------------------------------------------------------- #
class CustomMetricTester(BaseTester):
    def __init__(self, model, test_loader, eval_cfg, module_cfg, datamodule_cfg,
                 experiment_cfg, run_path, device='cpu', data_type='full', visualize=False):
        super().__init__(model, test_loader, eval_cfg, module_cfg, datamodule_cfg,
                         experiment_cfg, run_path, device, data_type, visualize)
        self.representative_slice = self.datamodule_cfg['metric']['representative_slice']
        self.stats_metadata_file = self.datamodule_cfg['stats_metadata_file']
        self.metadata_dict = load_meta(self.stats_metadata_file, self.representative_slice)
        self.class_names = [label for _, label in sorted(self.datamodule_cfg['mask_labels'].items())]
        self.tissue_labels = self.datamodule_cfg['metric']['tissues']
        self.metrics = load_metrics(self.datamodule_cfg['metric']['func'], self.class_names, self.tissue_labels)
        self.dicom_fields = self.datamodule_cfg['metric']['dicom_fields']

    def evaluate_multilabel_test_set(self):

        if self.module_cfg['visualize']:
            QCV.run_batch_quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():

            for batch_idx, batch in enumerate(self.test_loader):
                img_name = batch['img_name']
                images, gt2D = batch['image'].to(self.device), batch['gt2D'].to(self.device)
                boxes, label_ids = batch['boxes'].to(self.device), batch['label_ids'].to(self.device)
                
                # Determine the number of present classes for the current subject
                present_labels = torch.unique(gt2D[gt2D != 0]).tolist()  # Exclude background
                batch_size, height, width = gt2D.size(0), gt2D.size(2), gt2D.size(3)
                combined_mask = torch.zeros((batch_size, height, width), 
                                            device=self.device, dtype=torch.int64)

                for img_idx in range(batch_size):
                    subj_id, slice_id = parse_image_name(img_name[img_idx]) 

                    if self.representative_slice:
                        subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields)
                    else:
                        subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields, slice_id=slice_id)

                    for class_label in present_labels:
                        class_mask = torch.zeros((height, width), device=self.device)
                        class_boxes = boxes[img_idx][label_ids[img_idx] == class_label]
                        
                        for instance_idx, box in enumerate(class_boxes):
                            if box.sum() == 0:  # Skip if the box is all zeros (i.e., no valid box)
                                continue
                        
                            prediction = self.model(
                                images[img_idx].unsqueeze(0), 
                                box.unsqueeze(0)
                            )   

                            prediction_binary = torch.sigmoid(prediction) > 0.5  # Convert logits to binary predictions

                            # Update class_mask with max values
                            class_mask += prediction_binary.squeeze().float()

                        combined_mask[img_idx][class_mask > 0] = class_label
  
                if self.visualize and batch_idx % 5==0 and self.data_type == 'sampled': 
                    QCV.plot_gt_pred_mask_overlay(
                        image=images[0].detach().cpu(),
                        gt_mask=gt2D[0].squeeze(0).detach().cpu(),
                        pred_mask=combined_mask.squeeze(0).detach().cpu(),
                        mask_labels=self.datamodule_cfg['mask_labels'],
                        image_name=f"{batch['img_name'][0]}_{batch_idx}_combined_mask",
                        model_save_path=self.model_save_path,
                    )

                num_classes_for_onehot = self.datamodule_cfg['num_classes'] 
                to_onehot = AsDiscrete(to_onehot=num_classes_for_onehot)

                # Apply the transform to convert multi-class predictions to one-hot format
                combined_mask_onehot = to_onehot(combined_mask)  
                local_gt2D_onehot = to_onehot(gt2D.squeeze(1))     

                combined_mask_onehot = combined_mask_onehot.unsqueeze(0)  
                local_gt2D_onehot = local_gt2D_onehot.unsqueeze(0) 

                t1rho, t2 = None, None
                if 't1rho' in batch:
                    t1rho = batch['t1rho']
                if 't2' in batch:
                    t2 = batch['t2']

                for metric in self.metrics:
                    metric_name = metric.__class__.__name__.lower()

                    if 't1rho' in metric_name:
                        metric.update(y_pred=combined_mask_onehot, y=local_gt2D_onehot,
                                      map_array=t1rho, subject_slice_meta=subject_slice_meta,
                                      subj_id=subj_id, slice_id=slice_id)
                    elif 't2' in metric_name:
                        metric.update(y_pred=combined_mask_onehot, y=local_gt2D_onehot,
                                      map_array=t2, subject_slice_meta=subject_slice_meta,
                                      subj_id=subj_id, slice_id=slice_id)
                    else:
                        metric.update(y_pred=combined_mask_onehot, y=local_gt2D_onehot,
                                      subject_slice_meta=subject_slice_meta,
                                      subj_id=subj_id, slice_id=slice_id)

            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                total_volumes_pred, total_volumes_true = metric.compute()
                aggregated_volumes_pred, aggregated_volumes_true = metric.aggregate_by_subject()
                self.save_custom_metric_scores(metric_name, total_volumes_pred,
                                            total_volumes_true,
                                            aggregated_volumes_pred,
                                            aggregated_volumes_true)
                if self.module_cfg.get('use_wandb', False):
                    wandb_log(aggregated_volumes_pred)

    def evaluate_non_multilabel_test_set(self):
        """ for datasets without multiple instances of labels """

        if self.module_cfg['visualize']:
            QCV.run_batch_quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                img_names = batch['img_name']
                images = batch['image'].to(self.device)
                gt2D = batch['gt2D'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                multi_masks = torch.zeros_like(gt2D[0], device=self.device)
                num_classes = self.datamodule_cfg['num_classes']
                batch_size, height, width = multi_masks.size(0), multi_masks.size(1), multi_masks.size(2)
                multi_class_probs = torch.zeros((batch_size, num_classes, height, width), device=self.device)

                for img_idx in range(batch_size):
                    subj_id, slice_id = parse_image_name(img_names[img_idx])
                    if self.representative_slice:
                        subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields)
                    else:
                        subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields, slice_id=slice_id)
                    
                    for box, label_id in zip(boxes[0], label_ids[0]):
                        prediction = self.model(images, box.unsqueeze(0))
                        prediction_binary = torch.sigmoid(prediction) > 0.5
                        class_index = label_id
                        multi_class_probs[:, class_index, :, :] = torch.max(
                            multi_class_probs[:, class_index, :, :],
                            prediction_binary.float()
                        )
                    multi_masks = multi_class_probs.argmax(dim=1)
  
                    if self.visualize and batch_idx % 10==0 and self.data_type == 'sampled': 
                        QCV.plot_gt_pred_mask_overlay(
                            image=images[0].detach().cpu(),
                            gt_mask=gt2D[0].squeeze(0).detach().cpu(),
                            pred_mask=multi_masks.squeeze(0).detach().cpu(),
                            mask_labels=self.datamodule_cfg['mask_labels'],
                            image_name=f"{batch['img_name'][0]}_{batch_idx}_combined_mask",
                            model_save_path=self.model_save_path,
                        )

                    num_classes_for_onehot = self.datamodule_cfg['num_classes']
                    to_onehot = AsDiscrete(to_onehot=num_classes_for_onehot)
                    
                    multi_masks_onehot = to_onehot(multi_masks)
                    gt2D_onehot = to_onehot(gt2D.squeeze(1))
                    multi_masks_onehot = multi_masks_onehot.unsqueeze(0)
                    gt2D_onehot = gt2D_onehot.unsqueeze(0)

                    t1rho, t2 = None, None
                    if 't1rho' in batch:
                        t1rho = batch['t1rho']
                    if 't2' in batch:
                        t2 = batch['t2']

                    for metric in self.metrics:
                        metric_name = metric.__class__.__name__.lower()
                        if 't1rho' in metric_name:
                            metric.update(y_pred=multi_masks_onehot, y=gt2D_onehot,
                                          map_array=t1rho,
                                          subject_slice_meta=subject_slice_meta,
                                          subj_id=subj_id, slice_id=slice_id)
                        elif 't2' in metric_name:
                            metric.update(y_pred=multi_masks_onehot, y=gt2D_onehot,
                                          map_array=t2,
                                          subject_slice_meta=subject_slice_meta,
                                          subj_id=subj_id, slice_id=slice_id)
                        else:
                            metric.update(y_pred=multi_masks_onehot, y=gt2D_onehot,
                                          subject_slice_meta=subject_slice_meta,
                                          subj_id=subj_id, slice_id=slice_id)

            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                total_volumes_pred, total_volumes_true = metric.compute()
                aggregated_volumes_pred, aggregated_volumes_true = metric.aggregate_by_subject()
                self.save_custom_metric_scores(metric_name, total_volumes_pred,
                                            total_volumes_true,
                                            aggregated_volumes_pred,
                                            aggregated_volumes_true)
                if self.module_cfg.get('use_wandb', False):
                    wandb_log(aggregated_volumes_pred)

    @main_process_only
    def save_to_csv(self, data, path):
        # Check if data values are dictionaries or lists
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            all_columns = set()
            for subj_data in data.values():
                all_columns.update(subj_data.keys())
            all_columns = sorted(all_columns)
            df = pd.DataFrame.from_dict(data, orient='index')
            df = df.reindex(columns=all_columns)
        elif isinstance(first_val, list):
            df = pd.DataFrame.from_dict(data, orient='index')
            if len(df.columns) == len(self.class_names):
                df.columns = self.class_names
            elif len(df.columns) == len(self.class_names) + 1:
                df.columns = self.class_names + ['total']
            else:
                df.columns = [f"Label_{i}" for i in range(len(df.columns))]
        else:
            raise ValueError("Data structure not supported. Expected dict of dicts or dict of lists.")
        df.index.name = 'Subject'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"Saved custom metric scores to {path}")

    @main_process_only
    def save_custom_metric_scores(self, metric_name, total_volumes_pred, total_volumes_true,
                                  aggregated_volumes_pred, aggregated_volumes_true):
        base_path = os.path.join(self.run_path, 'test_eval')
        os.makedirs(base_path, exist_ok=True)
        filenames = [
            (f"{metric_name}_pred_slice_volume.csv", total_volumes_pred),
            (f"{metric_name}_gt_slice_volume.csv", total_volumes_true),
            (f"{metric_name}_pred_volume.csv", aggregated_volumes_pred),
            (f"{metric_name}_gt_volume.csv", aggregated_volumes_true)
        ]
        for filename, data in filenames:
            path = os.path.join(base_path, f"{self.run_id}_{self.datamodule_cfg['dataset_name']}_model_{self.eval_cfg['model_identifier']}", self.data_type, filename)
            self.save_to_csv(data, path)
            if self.module_cfg.get('use_wandb', False):
                wandb_log({filename: path})

    def test(self):
        log_info(f"Testing with {self.data_type} data...")
        if self.instance_bbox:
            self.evaluate_multilabel_test_set()
        else:
            self.evaluate_non_multilabel_test_set()

