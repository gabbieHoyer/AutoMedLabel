import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.distributed as dist
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete

from src.utils import QCV
import src.finetuning.utils.gpu_setup as GPUSetup 
from src.finetuning.utils.logging import main_process_only, log_info, wandb_log

logger = logging.getLogger(__name__)

# ------------------ Evaluation Base ------------------ #

# Base class for shared evaluation functionality
class BaseTester:
    def __init__(self, model, test_loader, eval_cfg, module_cfg,
                 datamodule_cfg, experiment_cfg, run_path, device='cpu',
                 data_type='full', visualize=False):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.eval_cfg = eval_cfg
        self.module_cfg = module_cfg
        self.datamodule_cfg = datamodule_cfg
        self.experiment_cfg = experiment_cfg
        self.run_path = run_path
        self.device = device
        self.data_type = data_type
        self.visualize = visualize

        self.num_classes = self.datamodule_cfg['num_classes']
        self.instance_bbox = self.datamodule_cfg['instance_bbox']

        self.model_save_path, self.run_id = self.setup_experiment_environment()

        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean_batch", ignore_empty=True,
            get_not_nans=False, num_classes=self.num_classes
        )
        self.IoU_metric = MeanIoU(
            include_background=False, reduction="mean_batch", ignore_empty=True,
            get_not_nans=False
        )

        self.shared_metrics = {
            'test_dice': [],
            'test_IoU': [],
            'class_dice_scores': {f'dice_{label}': [] for label in self.datamodule_cfg['mask_labels'].values()},
            'class_IoU_scores': {f'IoU_{label}': [] for label in self.datamodule_cfg['mask_labels'].values()}
        }
        self.slice_dice_metrics = []
        self.slice_IoU_metrics = []

    def setup_experiment_environment(self):
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        if self.module_cfg.get('use_wandb', False) and GPUSetup.is_main_process():
            import wandb
            wandb.login()
            wandb.init(
                project=self.module_cfg['task_name'],
                config={
                    "model_type": self.eval_cfg['model_type'],
                    "description": self.experiment_cfg['description'],
                    "finetuned_weights": self.eval_cfg['model_identifier'],
                    "model_path": self.eval_cfg['finetuned_model'],
                    "model_ID": os.path.basename(self.eval_cfg['finetuned_model']).split('_')[0],
                    "balanced": self.eval_cfg['model_details']['balanced'],
                    "subject_subject_set": self.eval_cfg['model_details']['subject'],
                    "image_encoder": self.eval_cfg['trainable']['image_encoder'],
                    "mask_decoder": self.eval_cfg['trainable']['mask_decoder'],
                    "bbox_shift": self.eval_cfg['model_details']['bbox_shift'],
                    "dataset": self.datamodule_cfg['dataset_name'],
                    "num_labels": self.num_classes,
                    "mask_labels": self.datamodule_cfg['mask_labels'],
                    "remove_label_ids": self.datamodule_cfg['remove_label_ids'],
                    "test_set_type": self.data_type
                },
                settings=wandb.Settings(_service_wait=300),
                tags=['test', self.experiment_cfg['name'], self.datamodule_cfg['dataset_name'],
                      self.eval_cfg['model_identifier']],
                name=run_id
            )
        return model_save_path, run_id

    def compute_batch_metrics(self, pred_mask, gt_mask):
        to_onehot = AsDiscrete(to_onehot=self.num_classes)
        pred_onehot = to_onehot(pred_mask)
        gt_onehot = to_onehot(gt_mask.squeeze(1))
        # Add a batch dimension if necessary
        pred_onehot = pred_onehot.unsqueeze(0)
        gt_onehot = gt_onehot.unsqueeze(0)
        dice_score = self.dice_metric(y_pred=pred_onehot, y=gt_onehot)
        IoU_score = self.IoU_metric(y_pred=pred_onehot, y=gt_onehot)
        return dice_score, IoU_score

    def update_slice_metrics(self, dice_score, IoU_score, img_name):
        dice_list = dice_score.tolist()
        IoU_list = IoU_score.tolist()
        slice_entry = {'filename': img_name[0]}
        IoU_slice_entry = {'filename': img_name[0]}

        for class_idx, score in enumerate(dice_list[0]):
            label_name = self.datamodule_cfg['mask_labels'].get(class_idx + 1, f'Class_{class_idx + 1}')
            slice_entry[label_name] = score
        for class_idx, score in enumerate(IoU_list[0]):
            label_name = self.datamodule_cfg['mask_labels'].get(class_idx + 1, f'Class_{class_idx + 1}')
            IoU_slice_entry[label_name] = score

        self.slice_dice_metrics.append(slice_entry)
        self.slice_IoU_metrics.append(IoU_slice_entry)

    def aggregate_and_log_global_metrics(self):
        dice_tensor = self.dice_metric.aggregate()
        IoU_tensor = self.IoU_metric.aggregate()

        if GPUSetup.is_distributed():
            dist.all_reduce(dice_tensor, op=dist.ReduceOp.SUM)
            dice_tensor /= (dist.get_world_size() * dice_tensor.numel())
        
        global_dice = dice_tensor.mean().item()
        global_IoU = IoU_tensor.mean().item()
        dice_log = {'global_dice_score': global_dice}
        IoU_log = {'global_IoU_score': global_IoU}
        mask_labels = self.datamodule_cfg['mask_labels']

        for i, score in enumerate(dice_tensor.tolist()):
            label_name = mask_labels.get(i + 1, f'Class_{i + 1}')
            dice_log[f'dice_{label_name}'] = score
            self.shared_metrics['class_dice_scores'][f'dice_{label_name}'].append(score)
        for i, score in enumerate(IoU_tensor.tolist()):
            label_name = mask_labels.get(i + 1, f'Class_{i + 1}')
            IoU_log[f'IoU_{label_name}'] = score
            self.shared_metrics['class_IoU_scores'][f'IoU_{label_name}'].append(score)

        if self.module_cfg.get('use_wandb', False):
            wandb_log(dice_log)
            wandb_log(IoU_log)

        self.shared_metrics['test_dice'].append(global_dice)
        self.shared_metrics['test_IoU'].append(global_IoU)
        log_info(f"Global average Dice score on test set: {global_dice:.4f}")
        log_info(f"Global average IoU score on test set: {global_IoU:.4f}")

    def evaluate_multilabel_test_set(self):
        # Default evaluation for multilabel datasets
        if self.module_cfg['visualize']:
            QCV.run_batch_quality_check(self.test_loader, self.model_save_path)
            # quality_check(self.test_loader, self.model_save_path)

        self.model.eval()
        self.dice_metric.reset()
        self.IoU_metric.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Implementation for obtaining predictions and metrics
                # ...
                # Example: use self.compute_batch_metrics and self.update_slice_metrics
                pass
            self.aggregate_and_log_global_metrics()

    @main_process_only
    def save_test_results(self):
        metrics_file_path = os.path.join(self.model_save_path, 'test_eval', f"{self.run_id}_{self.data_type}_metrics.json")
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        with open(metrics_file_path, 'w') as f:
            json.dump(self.shared_metrics, f, indent=4)

    @main_process_only
    def save_metric_scores_to_csv(self, metric_name, slice_metric):
        base_path = os.path.join(self.run_path, 'test_eval', self.data_type)
        os.makedirs(base_path, exist_ok=True)

        df = pd.DataFrame(slice_metric)
        slice_csv_path = os.path.join(base_path, f"slice_{metric_name}.csv")
        df.to_csv(slice_csv_path, index=False)
        print(f'Saved slice-level {metric_name} scores to {slice_csv_path}')
        
        df['volume_name'] = df['filename'].apply(lambda x: x.rsplit('-', 1)[0])
        aggregation_functions = {col: 'mean' for col in df.columns if col not in ['filename', 'volume_name']}
        aggregation_functions['filename'] = 'count'
        aggregated_df = df.groupby('volume_name').agg(aggregation_functions)
        aggregated_df.rename(columns={'filename': 'num_slices'}, inplace=True)
        aggregated_df.reset_index(inplace=True)
        volume_csv_path = os.path.join(base_path, f"volume_{metric_name}.csv")
        aggregated_df.to_csv(volume_csv_path, index=False)
        print(f'Saved volume-level {metric_name} scores to {volume_csv_path}')
        
        if 'num_slices' in aggregated_df.columns:
            aggregated_df.drop('num_slices', axis=1, inplace=True)
        numeric_cols = aggregated_df.select_dtypes(include=[np.number]).columns.tolist()
        global_scores = aggregated_df[numeric_cols].mean()
        num_volumes = len(df['volume_name'].unique())
        overall_global_metric = global_scores.mean()
        result_df = pd.DataFrame([global_scores], index=[0])
        result_df.insert(0, 'num_volumes', num_volumes)
        result_df[f'overall_global_{metric_name}'] = overall_global_metric
        global_csv_path = os.path.join(base_path, f"global_{metric_name}.csv")
        result_df.to_csv(global_csv_path, index=False)
        print(f'Saved global-level {metric_name} scores to {global_csv_path}')
        
        if self.module_cfg.get('use_wandb', False):
            wandb_log_data = {
                f"{metric_name}_slice_csv_path": os.path.join(root, slice_csv_path),
                f"{metric_name}_volume_csv_path": os.path.join(root, volume_csv_path),
                f"{metric_name}_global_csv_path": os.path.join(root, global_csv_path)
            }
            wandb_log(wandb_log_data)

    def test(self):
        log_info(f"Testing with {self.data_type} data...")
        if self.instance_bbox:
            self.evaluate_multilabel_test_set()
        else:
            self.evaluate_non_multilabel_test_set()
        self.save_test_results()

        if self.data_type == "full":
            self.save_metric_scores_to_csv(metric_name='dice', slice_metric=self.slice_dice_metrics)
            self.save_metric_scores_to_csv(metric_name='IoU', slice_metric=self.slice_IoU_metrics)
