import logging
from typing import List, Tuple, Dict, Optional, Any
import torch
from torch.utils.data import ConcatDataset

import src.finetuning.utils.gpu_setup as GPUSetup
from .base_datamodule import BaseDataModule
from .components.npy_dataset import MultiClassSAM2Dataset

logger = logging.getLogger(__name__)

class EvaluationDataModule(BaseDataModule):
    """
    DataModule for evaluation.
    Processes only the test split.
    Accepts an optional parameter metric_type to determine if biomarker paths are needed.
    """
    def __init__(self, dataset_cfg: Dict, metric_type: Optional[str] = None, **kwargs):
        super().__init__(dataset_cfg, **kwargs)
        self.splits_to_process = ['test']
        self.metric_type = metric_type
        # By default, no additional paths are passed.
        self.additional = None

    def create_dataset(self, 
                       paths: Dict[str, List[str]], 
                       mask_labels, 
                       instance_bbox, 
                       remove_label_ids, 
                       split: str) -> Any:
        # For the standard evaluation pipeline, simply pass the basic keys.
        dataset_args = {
            'root_paths': paths['root_paths'],
            'gt_paths': paths['gt_paths'],
            'img_paths': paths['img_paths'],
            'bbox_shift': self.bbox_shift,
            'mask_labels': mask_labels,
            'instance_bbox': instance_bbox,
            'remove_label_ids': remove_label_ids
        }
        return MultiClassSAM2Dataset(**dataset_args)
    
    def process_dataset(self, dataset_info: List[Tuple]) -> Tuple[Any, Any]:
        # Save original max_subjects
        original_max = self.max_subjects

        # For full dataset, force no filtering:
        self.max_subjects = 'full'
        full_results = self._process_splits_common(dataset_info, self.splits_to_process, additional=self.additional)
        aggregated_full = [info['datasets']['test'] for info in full_results.values()]
        full_test_dataset = ConcatDataset(aggregated_full) if len(aggregated_full) > 1 else aggregated_full[0]

        # Restore the original value for sampled dataset
        self.max_subjects = original_max
        if isinstance(self.max_subjects, int):
            sampled_results = self._process_splits_common(dataset_info, self.splits_to_process, additional=self.additional)
            aggregated_sampled = [info['datasets']['test'] for info in sampled_results.values()]
            sampled_test_dataset = (ConcatDataset(aggregated_sampled)
                                    if len(aggregated_sampled) > 1 else aggregated_sampled[0])
        else:
            sampled_test_dataset = None

        return sampled_test_dataset, full_test_dataset

    def get_loaders(self, batch_size: int, num_workers: int, instance_bbox: bool) -> dict:
        dataset_info = self.get_dataset_info()
        sampled_dataset, full_dataset = self.process_dataset(dataset_info)
        loaders = self.create_dataloader((sampled_dataset, full_dataset), batch_size, num_workers, instance_bbox)
        return {"loaders": {"sampled": loaders[0], "full": loaders[1]}}



class BiomarkerEvaluationDataModule(EvaluationDataModule):
    """
    DataModule for biomarker evaluation.
    This subclass adds biomarker-specific dataset arguments
    and uses a specialized collate function that stacks t1rho and t2 maps.
    """
    def __init__(self, dataset_cfg: dict, metric_type: dict = None, **kwargs):
        super().__init__(dataset_cfg, metric_type=metric_type, **kwargs)

        # metric_type might be None or a dict like {"t1rho": True, "t2": True}.
        # store it as a dict for direct checks.
        self.metric_type = metric_type or {}

        # If either t1rho or t2 is enabled, we will want additional paths.
        # (You could rename these to something else, e.g. self.use_biomarkers.)
        self.use_t1rho = bool(self.metric_type.get('t1rho', False))
        self.use_t2 = bool(self.metric_type.get('t2', False))

        # If either is True, we want T1rho_map_path / T2_map_path.
        if self.use_t1rho or self.use_t2:
            self.additional = ['T1rho_map_path', 'T2_map_path']
        else:
            self.additional = None

    def create_dataset(self, paths: dict, mask_labels, instance_bbox, remove_label_ids, split: str):
        # Use the base keys
        dataset_args = {
            'root_paths': paths['root_paths'],
            'gt_paths': paths['gt_paths'],
            'img_paths': paths['img_paths'],
            'bbox_shift': self.bbox_shift,
            'mask_labels': mask_labels,
            'instance_bbox': instance_bbox,
            'remove_label_ids': remove_label_ids
        }

        # If t1rho or t2 is enabled, attach extra map paths.
        if self.use_t1rho or self.use_t2:
            dataset_args['use_biomarkers'] = True
            if self.use_t1rho:
                dataset_args['T1rho_map_paths'] = paths.get('T1rho_map_path')
            if self.use_t2:
                dataset_args['T2_map_paths'] = paths.get('T2_map_path')

        return MultiClassSAM2Dataset(**dataset_args)

    def create_dataloader(self, datasets, batch_size, num_workers, instance_bbox: bool = False):
        from torch.utils.data import DataLoader, DistributedSampler
        """
        Override the dataloader creation to use a specialized collate_fn if needed.
        """
        num_tasks = GPUSetup.get_world_size()
        global_rank = GPUSetup.get_rank()
        # For evaluation, we expect datasets to be a tuple of two: (sampled, full)
        if isinstance(datasets, (list, tuple)) and len(datasets) == 2:
            sampled_dataset, full_dataset = datasets
          
            # If instance_bbox is True, use our special collate_fn that also can stack T1/T2 maps if they exist.
            if instance_bbox:
                collate_function = lambda batch: self.collate_fn(
                    batch,
                    use_biomarkers=(self.use_t1rho or self.use_t2)
                )
            else:
                collate_function = None

            sampled_loader = DataLoader(
                sampled_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_function
            )
            full_loader = DataLoader(
                full_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_function
            )
            logger.info(f"Rank {global_rank}: DataLoaders (sampled, full) initialized with batch_size={batch_size}, num_workers={num_workers}")
            return sampled_loader, full_loader
        else:
            return super().create_dataloader(datasets, batch_size, num_workers, instance_bbox)

    def collate_fn(self, batch, use_biomarkers=False):
        images = [item['image'] for item in batch]
        gt2D = [item['gt2D'] for item in batch]
        boxes = [item['boxes'] for item in batch]
        label_ids = [item['label_ids'] for item in batch]
        img_names = [item['img_name'] for item in batch]
        
        images = torch.stack(images)
        gt2D = torch.stack(gt2D)

        if use_biomarkers:
            t1rho_maps = torch.stack([item['t1rho'] for item in batch])
            t2_maps = torch.stack([item['t2'] for item in batch])
        
        max_num_boxes = max(box.shape[0] for box in boxes)
        padded_boxes = []
        padded_labels = []
        for box, label in zip(boxes, label_ids):
            num_boxes = box.shape[0]
            padded_box = torch.zeros((max_num_boxes, 4))
            padded_label = torch.zeros((max_num_boxes,))
            if num_boxes > 0:
                padded_box[:num_boxes, :] = box
                padded_label[:num_boxes] = label
            padded_boxes.append(padded_box)
            padded_labels.append(padded_label)

        padded_boxes = torch.stack(padded_boxes)
        padded_labels = torch.stack(padded_labels)

        batch_data = {
            'image': images,
            'gt2D': gt2D,
            'boxes': padded_boxes,
            'label_ids': padded_labels,
            'img_name': img_names
        }
        if use_biomarkers:
            batch_data.update({
                't1rho': t1rho_maps,
                't2': t2_maps
            })
        return batch_data