import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import monai
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import src.finetuning.utils.gpu_setup as GPUSetup #is_distributed
from src.finetuning.utils.logging import main_process_only, log_info, wandb_log
from src.finetuning.utils.fig_QC import quality_check, visualize_input, visualize_predictions
from src.finetuning.utils.metric_utils import load_meta, parse_image_name, extract_meta
from src.finetuning.engine.metrics.metric_factory import load_metrics

logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, model, test_loader, eval_cfg, module_cfg, datamodule_cfg, experiment_cfg, run_path, device='cpu', data_type='full', visualize=False):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.eval_cfg = eval_cfg
        self.module_cfg = module_cfg
        self.datamodule_cfg = datamodule_cfg
        self.experiment_cfg = experiment_cfg
        self.run_path = run_path
        self.device = device
        self.data_type = data_type  # 'sampled' or 'full'
        self.visualize = visualize

        self.model_save_path, self.run_id = self.setup_experiment_environment()

        self.num_classes = self.datamodule_cfg['num_classes']
        self.instance_bbox = self.datamodule_cfg['instance_bbox']
        self.representative_slice = self.datamodule_cfg['metric']['representative_slice']
        self.stats_metadata_file = self.datamodule_cfg['stats_metadata_file']
        self.metadata_dict = load_meta(self.stats_metadata_file, self.representative_slice)

        self.class_names = [label for _, label in sorted(self.datamodule_cfg['mask_labels'].items())]
        self.tissue_labels = self.datamodule_cfg['metric']['tissues']

        # Load multiple metrics using the factory
        self.metrics = load_metrics(self.datamodule_cfg['metric']['func'], self.class_names, self.tissue_labels)

        self.dicom_fields = self.datamodule_cfg['metric']['dicom_fields']


    def setup_experiment_environment(self):
        # Use the already determined run_path
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")

        # Initialize Weights & Biases if configured
        if self.module_cfg.get('use_wandb', False) and GPUSetup.is_main_process():
            import wandb
            wandb.login()
            wandb.init(project=self.module_cfg['task_name'], 
                    config={
                        "model_type": self.eval_cfg['model_type'],
                        "description": self.experiment_cfg['description'],
                        "finetuned_weights": self.eval_cfg['model_weights'],
                        "balanced": self.eval_cfg['model_details']['balanced'],
                        "subject_subject_set": self.eval_cfg['model_details']['subject'],
                        "image_encoder": self.eval_cfg['trainable']['image_encoder'],
                        "mask_decoder": self.eval_cfg['trainable']['mask_decoder'],
                        "bbox_shift": self.eval_cfg['model_details']['bbox_shift'],
                        "dataset": self.datamodule_cfg['dataset_name'],
                        "num_labels": self.datamodule_cfg['num_classes'],
                        "mask_labels": self.datamodule_cfg['mask_labels'],
                        "remove_label_ids": self.datamodule_cfg['remove_label_ids'],
                        "test_set_type": self.data_type
                    }, 
                    settings=wandb.Settings(_service_wait=300),
                    tags=['test', self.experiment_cfg['name'], self.datamodule_cfg['dataset_name'], self.eval_cfg['model_weights']],
                    name="{}_{}_{}".format(self.datamodule_cfg['dataset_name'], self.eval_cfg['model_weights'], run_id)
                    )
        return model_save_path, run_id
    
    def evaluate_multilabel_test_set(self):
        """ for datasets with multiple instances of labels """

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

                batch_size, height, width = images.size(0), images.size(2), images.size(3)
                combined_mask = torch.zeros((batch_size, height, width), device=self.device, dtype=torch.int64)

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
                        
                            # Assuming your model returns predictions for the full image when given a box
                            prediction = self.model(images[img_idx].unsqueeze(0), box.unsqueeze(0))   

                            prediction_binary = torch.sigmoid(prediction) > 0.5  # Convert logits to binary predictions

                            # Update class_mask with max values
                            class_mask += prediction_binary.squeeze().float()

                        combined_mask[img_idx][class_mask > 0] = class_label

                num_classes_for_onehot = self.datamodule_cfg['num_classes'] 
                as_discrete_transform = AsDiscrete(to_onehot=num_classes_for_onehot)

                # Apply the transform to convert multi-class predictions to one-hot format
                combined_mask_onehot = as_discrete_transform(combined_mask)  
                gt2D_squeezed = gt2D.squeeze(1)
                local_gt2D_onehot = as_discrete_transform(gt2D_squeezed)     

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

                # import pdb; pdb.set_trace()

                self.save_custom_metric_scores(metric_name, total_volumes_pred, total_volumes_true, aggregated_volumes_pred, aggregated_volumes_true)

                # in dev - hasn't been tested
                if self.module_cfg.get('use_wandb', False):
                    wandb_log(aggregated_volumes_pred)  # Log all scores to wandb


    def evaluate_non_multilabel_test_set(self):
        """ for datasets without multiple instances of labels """

        if self.module_cfg['visualize'] and self.data_type == 'sampled':
            quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        for metric in self.metrics:
            metric.reset()
        
        with torch.no_grad():

            for batch_idx, batch in enumerate(self.test_loader):
                img_name = batch['img_name']
                images, gt2D = batch['image'].to(self.device), batch['gt2D'].to(self.device)
                boxes, label_ids = batch['boxes'].to(self.device), batch['label_ids'].to(self.device)
                
                multi_masks = torch.zeros_like(gt2D[0], device=self.device)
                num_classes = self.datamodule_cfg['num_classes']  

                batch_size, height, width = multi_masks.size(0), multi_masks.size(1), multi_masks.size(2)
                multi_class_probs = torch.zeros((batch_size, num_classes, height, width), device=self.device)

                for img_idx in range(batch_size):

                    subj_id, slice_id = parse_image_name(img_name[img_idx]) 

                    if self.representative_slice:
                        subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields)
                    else:
                        subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields, slice_id=slice_id)

                for box, label_id in zip(boxes[0], label_ids[0]):  
                    # Get prediction for a single bounding box
                    prediction = self.model(images, box.unsqueeze(0))  

                    # prediction_probs = torch.sigmoid(prediction)  # Convert logits to probabilities
                    prediction_binary = torch.sigmoid(prediction) > 0.5  # Convert logits to binary predictions

                    class_index = label_id

                    multi_class_probs[:, class_index, :, :] = torch.max(
                        multi_class_probs[:, class_index, :, :], prediction_binary.float()
                    )

                # Now we select the label with the highest probability for each pixel
                multi_masks = multi_class_probs.argmax(dim=1)

                num_classes_for_onehot = self.datamodule_cfg['num_classes'] 
                as_discrete_transform = AsDiscrete(to_onehot=num_classes_for_onehot)

                # Apply the transform to convert multi-class predictions to one-hot format
                multi_masks_onehot = as_discrete_transform(multi_masks)  
                gt2D_squeezed = gt2D.squeeze(1)
                gt2D_onehot = as_discrete_transform(gt2D_squeezed)     

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
                                      map_array=t1rho, subject_slice_meta=subject_slice_meta,
                                      subj_id=subj_id, slice_id=slice_id)
                    elif 't2' in metric_name:
                        metric.update(y_pred=multi_masks_onehot, y=gt2D_onehot,
                                      map_array=t2, subject_slice_meta=subject_slice_meta,
                                      subj_id=subj_id, slice_id=slice_id)
                    else:
                        metric.update(y_pred=multi_masks_onehot, y=gt2D_onehot,
                                      subject_slice_meta=subject_slice_meta,
                                      subj_id=subj_id, slice_id=slice_id)

            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()

                total_volumes_pred, total_volumes_true = metric.compute()
                aggregated_volumes_pred, aggregated_volumes_true = metric.aggregate_by_subject()

                # import pdb; pdb.set_trace()

                self.save_custom_metric_scores(metric_name, total_volumes_pred, total_volumes_true, aggregated_volumes_pred, aggregated_volumes_true)

                # in dev - hasn't been tested
                if self.module_cfg.get('use_wandb', False):
                    wandb_log(aggregated_volumes_pred)  # Log all scores to wandb

    @main_process_only
    def save_to_csv(self, data, path):
        df = pd.DataFrame.from_dict(data, orient='index')
        if len(df.columns) == len(self.class_names) + 1:
            df.columns = self.class_names + ['total']

        # likely  need more complex logic to handle case of instance labels vs not
        # how are these lining up exactly????
        else:
            df.columns = self.class_names

        df.index.name = 'Subject'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f'Saved custom metric scores to {path}')

    def save_custom_metric_scores(self, metric_name, total_volumes_pred, total_volumes_true, aggregated_volumes_pred, aggregated_volumes_true):
        base_path = os.path.join(self.run_path, 'test_eval')
        os.makedirs(base_path, exist_ok=True)
        
        filenames = [
            (f"{metric_name}_pred_slice_volume.csv", total_volumes_pred),
            (f"{metric_name}_gt_slice_volume.csv", total_volumes_true),
            (f"{metric_name}_pred_volume.csv", aggregated_volumes_pred),
            (f"{metric_name}_gt_volume.csv", aggregated_volumes_true)
        ]
        
        for filename, data in filenames:
            path = os.path.join(base_path, f"{self.run_id}_{self.data_type}_data_{self.datamodule_cfg['dataset_name']}_model_{self.eval_cfg['model_weights']}", filename)
            self.save_to_csv(data, path)

    def test(self):
        logger.info(f"Testing with {self.data_type} data...")
        
        if self.instance_bbox:
            self.evaluate_multilabel_test_set()
            # figure out way to match up label from dictionary to mask labels
            # so that they will include 'mask label' + 'instance_#' 
        else:
            self.evaluate_non_multilabel_test_set()

        # self.save_test_results()






    # @main_process_only
    # def save_to_csv(self, data, path):
    #     df = pd.DataFrame.from_dict(data, orient='index')
    #     df.columns = self.class_names + ['total']
    #     df.index.name = 'Subject'
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     df.to_csv(path)
    #     print(f'Saved custom metric scores to {path}')