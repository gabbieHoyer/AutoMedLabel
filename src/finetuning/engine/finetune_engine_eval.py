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

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True, get_not_nans=False, num_classes=datamodule_cfg['num_classes'])
        self.IoU_metric = MeanIoU(include_background=False, reduction="mean_batch", ignore_empty=True, get_not_nans=False)

        self.test_set_details = {'predictions': [], 'gt2D': []}
        # Initialize shared_metrics to store all relevant scores
        self.shared_metrics = {'test_dice': [], 'test_IoU': [], 
                               'class_dice_scores': {f'dice_{label}': [] for label in self.datamodule_cfg['mask_labels'].values()},
                               'class_IoU_scores': {f'IoU_{label}': [] for label in self.datamodule_cfg['mask_labels'].values()}
                               }

        # Add an attribute for aggregating slice-level results
        self.slice_dice_metrics = []
        self.slice_IoU_metrics = []

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
                        "model_ID": self.eval_cfg['finetuned_model'],
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

        if self.module_cfg['visualize'] and self.data_type == 'sampled':
            quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        self.dice_metric.reset()
        self.IoU_metric.reset() 

        with torch.no_grad():

            for batch_idx, batch in enumerate(self.test_loader):
                img_name = batch['img_name']
                images, gt2D, boxes, label_ids = batch['image'], batch['gt2D'], batch['boxes'], batch['label_ids']
                
                images, gt2D = images.to(self.device), gt2D.to(self.device)
                boxes, label_ids = boxes.to(self.device), label_ids.to(self.device)

                # Determine the number of present classes for the current subject
                present_labels = torch.unique(gt2D[gt2D != 0]).tolist()  # Exclude background

                batch_size, height, width = images.size(0), images.size(2), images.size(3)
                combined_mask = torch.zeros((batch_size, height, width), device=self.device, dtype=torch.int64)

                for img_idx in range(batch_size):
                    for class_label in present_labels:

                        class_mask = torch.zeros((height, width), device=self.device)
                        class_boxes = boxes[img_idx][label_ids[img_idx] == class_label]
                        
                        for instance_idx, box in enumerate(class_boxes):
                            if box.sum() == 0:  # Skip if the box is all zeros (i.e., no valid box)
                                continue
                        
                            # Assuming your model returns predictions for the full image when given a box
                            prediction = self.model(images[img_idx].unsqueeze(0), box.unsqueeze(0))   

                            # this works fine for instances per label
                            if self.module_cfg['visualize'] and batch_idx < 1 and self.data_type == 'sampled': 
                                gt_mask_single_label = (gt2D == class_label).float() * gt2D[0]
                                visualize_input(
                                    image=images[img_idx].detach().cpu(),                  
                                    gt_mask=gt_mask_single_label.detach().cpu().squeeze(),  
                                    box=box.detach().cpu(),  
                                    pred_mask=prediction.detach().cpu().squeeze(),
                                    label_id=class_label,  
                                    image_name=f"{img_name[img_idx]}_class_label_{class_label}_instance{instance_idx}", 
                                    model_save_path=self.model_save_path
                                )
                            
                            prediction_binary = torch.sigmoid(prediction) > 0.5  # Convert logits to binary predictions

                            # Update class_mask with max values
                            class_mask += prediction_binary.squeeze().float()

                        # combined_mask[img_idx][class_mask > 0] = local_label
                        combined_mask[img_idx][class_mask > 0] = class_label

                # Finally works! :D looks good, baby!
                if self.visualize and batch_idx < 2 and self.data_type == 'sampled':  # Visualize only for the first 5 batches
                    visualize_predictions(
                        image=images[0].cpu(),
                        gt_mask=gt2D[0].squeeze(0).cpu(),  
                        pred_mask=combined_mask.squeeze(0).cpu(),  
                        boxes=boxes[0].cpu(),
                        mask_labels=self.datamodule_cfg['mask_labels'],
                        image_name=f"{self.data_type}_{batch['img_name'][0]}_{batch_idx}", 
                        model_save_path=self.model_save_path
                    )

                num_classes_for_onehot = self.datamodule_cfg['num_classes'] 
                as_discrete_transform = AsDiscrete(to_onehot=num_classes_for_onehot)

                # Apply the transform to convert multi-class predictions to one-hot format
                combined_mask_onehot = as_discrete_transform(combined_mask)  
                gt2D_squeezed = gt2D.squeeze(1)
                local_gt2D_onehot = as_discrete_transform(gt2D_squeezed)     

                combined_mask_onehot = combined_mask_onehot.unsqueeze(0)  
                local_gt2D_onehot = local_gt2D_onehot.unsqueeze(0) 

                dice_score = self.dice_metric(y_pred=combined_mask_onehot, y=local_gt2D_onehot)
                IoU_score = self.IoU_metric(y_pred=combined_mask_onehot, y=local_gt2D_onehot)

                # Option to replace null instance values with an int like 0
                # dice_score = torch.nan_to_num(dice_score)
                # IoU_score = torch.nan_to_num(IoU_score)

                if self.data_type == "full":
                    # Convert tensor to list and iterate over each class's score
                    dice_scores_list = dice_score.tolist()
                    slice_entry = {'filename': img_name[0]} 
                    IoU_scores_list = IoU_score.tolist()
                    IoU_slice_entry = {'filename': img_name[0]} 

                    # Iterate over each class score and add to the slice entry
                    for class_idx, score in enumerate(dice_scores_list[0]):
                        label_name = self.datamodule_cfg['mask_labels'].get(class_idx + 1, f'Class_{class_idx + 1}')
                        slice_entry[label_name] = score  # Add each class score under its label as a column

                    for class_idx, score in enumerate(IoU_scores_list[0]):
                        label_name = self.datamodule_cfg['mask_labels'].get(class_idx + 1, f'Class_{class_idx + 1}')
                        IoU_slice_entry[label_name] = score  

                    # Append the structured slice entry to slice_metrics
                    self.slice_dice_metrics.append(slice_entry)
                    self.slice_IoU_metrics.append(IoU_slice_entry)

            dice_score_tensor = self.dice_metric.aggregate()  
            IoU_score_tensor = self.IoU_metric.aggregate()  

        if GPUSetup.is_distributed():  # might remove this capability fully
            # Sum each class's dice scores across GPUs
            dist.all_reduce(dice_score_tensor, op=dist.ReduceOp.SUM)
            # Compute the global mean across all GPUs and all classes
            dice_score_tensor /= (dist.get_world_size() * dice_score_tensor.numel())

        # Now we can safely convert to a float, since it's a scalar
        global_dice_score = dice_score_tensor.mean().item()
        global_IoU_score = IoU_score_tensor.mean().item()

        # Prepare to log individual class Dice scores and global Dice score
        dice_scores_log = {'global_dice_score': global_dice_score}
        IoU_scores_log = {'global_IoU_score': global_IoU_score}

        mask_labels = self.datamodule_cfg['mask_labels']
        for i, score in enumerate(dice_score_tensor.tolist()):
            label_name = mask_labels.get(i + 1, f'Class_{i + 1}')
            dice_scores_log[f'dice_{label_name}'] = score
            self.shared_metrics['class_dice_scores'][f'dice_{label_name}'].append(score)

        for i, score in enumerate(IoU_score_tensor.tolist()):
            label_name = mask_labels.get(i + 1, f'Class_{i + 1}')
            IoU_scores_log[f'IoU_{label_name}'] = score
            self.shared_metrics['class_IoU_scores'][f'IoU_{label_name}'].append(score)

        if self.module_cfg.get('use_wandb', False):
            wandb_log(dice_scores_log)  # Log all scores to wandb
            wandb_log(IoU_scores_log)

        self.shared_metrics['test_dice'].append(global_dice_score)
        self.shared_metrics['test_IoU'].append(global_IoU_score)

        log_info(f"Global average Dice score on test set: {global_dice_score:.4f}")
        log_info(f"Global average IoU score on test set: {global_IoU_score:.4f}")


    def evaluate_non_multilabel_test_set(self):
        """ for datasets without multiple instances of labels """

        if self.module_cfg['visualize'] and self.data_type == 'sampled':
            quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        self.dice_metric.reset()
        
        with torch.no_grad():

            for batch_idx, batch in enumerate(self.test_loader):
                img_name = batch['img_name']
                images, gt2D, boxes, label_ids = batch['image'], batch['gt2D'], batch['boxes'], batch['label_ids']
                 
                images, gt2D = images.to(self.device), gt2D.to(self.device)
                boxes, label_ids = boxes.to(self.device), label_ids.to(self.device)

                multi_masks = torch.zeros_like(gt2D[0], device=self.device)
                num_classes = self.datamodule_cfg['num_classes']  

                batch_size, height, width = multi_masks.size(0), multi_masks.size(1), multi_masks.size(2)
                multi_class_probs = torch.zeros((batch_size, num_classes, height, width), device=self.device)

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

                # Now both tensors should be in one-hot format and can be passed to the Dice metric
                dice_score = self.dice_metric(y_pred=multi_masks_onehot, y=gt2D_onehot)
                IoU_score = self.IoU_metric(y_pred=multi_masks_onehot, y=gt2D_onehot)

                # Option to replace null instance values with an int like 0
                # dice_score = torch.nan_to_num(dice_score)
                # IoU_score = torch.nan_to_num(IoU_score)

                if self.visualize and batch_idx < 2 and self.data_type == 'sampled':  # Visualize only for the first 5 batches
                    visualize_predictions(
                        image=images[0].cpu(),
                        gt_mask=gt2D[0].squeeze(0).cpu(),  
                        pred_mask=multi_masks.squeeze(0).cpu(),  
                        boxes=boxes[0].cpu(),
                        mask_labels=self.datamodule_cfg['mask_labels'],
                        image_name=f"{batch['img_name'][0]}_{batch_idx}",
                        model_save_path=self.model_save_path
                    )
                
                if self.data_type == "full":
                    # Convert tensor to list and iterate over each class's score
                    dice_scores_list = dice_score.tolist()
                    slice_entry = {'filename': img_name[0]} 
                    IoU_scores_list = IoU_score.tolist()
                    IoU_slice_entry = {'filename': img_name[0]} 

                    # Iterate over each class score and add to the slice entry
                    for class_idx, score in enumerate(dice_scores_list[0]):
                        label_name = self.datamodule_cfg['mask_labels'].get(class_idx + 1, f'Class_{class_idx + 1}')
                        slice_entry[label_name] = score  # Add each class score under its label as a column

                    for class_idx, score in enumerate(IoU_scores_list[0]):
                        label_name = self.datamodule_cfg['mask_labels'].get(class_idx + 1, f'Class_{class_idx + 1}')
                        IoU_slice_entry[label_name] = score  

                    # Append the structured slice entry to slice_metrics
                    self.slice_dice_metrics.append(slice_entry)
                    self.slice_IoU_metrics.append(IoU_slice_entry)

            dice_score_tensor = self.dice_metric.aggregate()  
            IoU_score_tensor = self.IoU_metric.aggregate()  

        if GPUSetup.is_distributed():
            # Sum each class's dice scores across GPUs
            dist.all_reduce(dice_score_tensor, op=dist.ReduceOp.SUM)
            # Compute the global mean across all GPUs and all classes
            dice_score_tensor /= (dist.get_world_size() * dice_score_tensor.numel())

        # Now we can safely convert to a float, since it's a scalar
        global_dice_score = dice_score_tensor.mean().item()
        global_IoU_score = IoU_score_tensor.mean().item()

        # Prepare to log individual class Dice scores and global Dice score
        dice_scores_log = {'global_dice_score': global_dice_score}
        IoU_scores_log = {'global_IoU_score': global_IoU_score}

        mask_labels = self.datamodule_cfg['mask_labels']
        for i, score in enumerate(dice_score_tensor.tolist()):
            label_name = mask_labels.get(i + 1, f'Class_{i + 1}')
            dice_scores_log[f'dice_{label_name}'] = score
            self.shared_metrics['class_dice_scores'][f'dice_{label_name}'].append(score)

        for i, score in enumerate(IoU_score_tensor.tolist()):
            label_name = mask_labels.get(i + 1, f'Class_{i + 1}')
            IoU_scores_log[f'IoU_{label_name}'] = score
            self.shared_metrics['class_IoU_scores'][f'IoU_{label_name}'].append(score)

        if self.module_cfg.get('use_wandb', False):
            wandb_log(dice_scores_log)  # Log all scores to wandb
            wandb_log(IoU_scores_log)

        self.shared_metrics['test_dice'].append(global_dice_score)
        self.shared_metrics['test_IoU'].append(global_IoU_score)

        log_info(f"Global average Dice score on test set: {global_dice_score:.4f}")
        log_info(f"Global average IoU score on test set: {global_IoU_score:.4f}")

    @main_process_only
    def save_test_results(self):
        metrics_file_path = os.path.join(self.model_save_path, 'test_eval', f"{self.run_id}_{self.data_type}_metrics.json")
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

        # Write the shared_metrics dictionary to a JSON file
        with open(metrics_file_path, 'w') as f:
            json.dump(self.shared_metrics, f, indent=4)  # Use indent for pretty printing
    
    @main_process_only
    def save_metric_scores_to_csv(self, metric_name, slice_metric):
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(slice_metric)
        slice_csv_path = os.path.join(self.run_path, 'test_eval', f"{self.run_id}_data_{self.datamodule_cfg['dataset_name']}_model_{self.eval_cfg['model_weights']}_slice_{metric_name}.csv")
        df.to_csv(slice_csv_path, index=False)
        print(f'Saved slice-level {metric_name} scores to {slice_csv_path}')

        # Process filenames to extract the volume base name
        df['volume_name'] = df['filename'].apply(lambda x: x.rsplit('-', 1)[0])
        
        # Define aggregation functions for each column
        # 'mean' will automatically skip NaN values
        aggregation_functions = {col: 'mean' for col in df.columns if col not in ['filename', 'volume_name']}
        aggregation_functions['filename'] = 'count'  # Count the number of slices for 'num_slices'
        
        # Aggregate the metrics
        aggregated_df = df.groupby('volume_name').agg(aggregation_functions)
        
        # Rename the 'filename' column to 'num_slices'
        aggregated_df.rename(columns={'filename': 'num_slices'}, inplace=True)
        
        # Prepare the output DataFrame
        aggregated_df.reset_index(inplace=True)
        
        # Save to new CSV
        volume_csv_path = os.path.join(self.run_path, 'test_eval', f"{self.run_id}_data_{self.datamodule_cfg['dataset_name']}_model_{self.eval_cfg['model_weights']}_volume_{metric_name}.csv")
        aggregated_df.to_csv(volume_csv_path, index=False)
        print(f'Saved volume-level {metric_name} scores to {volume_csv_path}')

        # Drop the 'num_slices' column from the DataFrame before processing
        if 'num_slices' in aggregated_df.columns:
            aggregated_df.drop('num_slices', axis=1, inplace=True)

        # Ensure only numeric columns are used for computing means
        numeric_cols = aggregated_df.select_dtypes(include=[np.number]).columns.tolist()
        global_scores = aggregated_df[numeric_cols].mean()

        num_volumes = len(df['volume_name'].unique())  # Count the number of unique volumes

        # Compute the overall global Dice score (average of all numeric columns' Dice scores)
        overall_global_metric = global_scores.mean()

        # Prepare the output DataFrame
        result_df = pd.DataFrame([global_scores], index=[0])
        result_df.insert(0, 'num_volumes', num_volumes)
        result_df[f'overall_global_{metric_name}'] = overall_global_metric

        # Save to new CSV
        global_csv_path = os.path.join(self.run_path, 'test_eval', f"{self.run_id}_data_{self.datamodule_cfg['dataset_name']}_model_{self.eval_cfg['model_weights']}_global_{metric_name}.csv")
        result_df.to_csv(global_csv_path, index=False)
        print(f'Saved global-level {metric_name} scores to {global_csv_path}')

        # Log all the paths to wandb at once
        if self.module_cfg.get('use_wandb', False):
            # Create the wandb_log_data dictionary at the bottom
            wandb_log_data = {
                "slice_csv_path": slice_csv_path,
                "volume_csv_path": volume_csv_path,
                "global_csv_path": global_csv_path
            }
            wandb_log(wandb_log_data)

    def test(self):
        logger.info(f"Testing with {self.data_type} data...")
        
        if self.instance_bbox:
            self.evaluate_multilabel_test_set()
        else:
            self.evaluate_non_multilabel_test_set()

        self.save_test_results()

        if self.data_type == "full":
            self.save_metric_scores_to_csv(metric_name='dice', slice_metric=self.slice_dice_metrics)
            self.save_metric_scores_to_csv(metric_name='IoU', slice_metric=self.slice_IoU_metrics)

