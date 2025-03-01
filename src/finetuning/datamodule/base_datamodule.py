
import logging
from typing import List, Tuple, Dict, Optional, Any
import torch

from src.utils import load_yaml
import src.finetuning.utils.gpu_setup as GPUSetup
from .components.experiment_summary import (
    load_and_process_splits_metadata,
    extract_paths_and_count_slices,
    filter_subjects_by_max_number,
)

logger = logging.getLogger(__name__)

class BaseDataModule:
    """
    BaseDataModule encapsulates shared functionality:
      - Reading dataset configuration and extracting metadata info.
      - Processing splits (train/val/test or test-only) into Dataset objects.
      - Providing a default collate_fn and creating DataLoaders.
    """
    def __init__(self, dataset_cfg: Dict, is_balanced: bool = False, bbox_shift: int = 0, 
                 max_subjects: Optional[int | str] = 'full'):
        self.dataset_cfg = dataset_cfg
        self.is_balanced = is_balanced
        self.bbox_shift = bbox_shift
        self.max_subjects = max_subjects
        self.logger = logger

    def get_dataset_info(self) -> List[Tuple]:
        """
        Reads the dataset configuration and returns a list of tuples
        containing dataset name, metadata file, parquet directory, mask labels,
        instance_bbox flag, removal IDs, and downsampling factors.
        """
        dataset_name_list = []
        ml_metadata_file_list = []
        slice_info_parquet_dir_list = []
        mask_labels_list = []
        instance_bbox_list = []
        remove_label_ids_list = []
        for dataset_name in self.dataset_cfg.keys():
            dataset_name_list.append(dataset_name)
            mask_labels_list.append(self.dataset_cfg[dataset_name].get('mask_labels'))
            instance_bbox_list.append(self.dataset_cfg[dataset_name].get('instance_bbox'))
            remove_label_ids_list.append(self.dataset_cfg[dataset_name].get('remove_label_ids'))
            ml_metadata_file_list.append(self.dataset_cfg[dataset_name].get('ml_metadata_file'))
            slice_info_parquet_dir_list.append(self.dataset_cfg[dataset_name].get('slice_info_parquet_dir'))
        if self.is_balanced:
            self.logger.info("Calculating downsampling factors for balanced loading")
            downsampling_factors = self.calculate_downsampling_factors(ml_metadata_file_list)
        else:
            self.logger.info("Balanced loading not enabled, proceeding without downsampling")
            downsampling_factors = [1] * len(dataset_name_list)
        dataset_info = list(zip(dataset_name_list, ml_metadata_file_list, slice_info_parquet_dir_list,
                                  mask_labels_list, instance_bbox_list, remove_label_ids_list,
                                  downsampling_factors))
        return dataset_info

    def calculate_downsampling_factors(self, metadata_paths: List[str]) -> List[int]:
        """Calculate downsampling factors based on total number of slices in each dataset."""
        num_slices_per_dataset = []
        for path in metadata_paths:
            config = load_yaml(path)
            total_slices = sum(subject['num_slices'] for subject in config.values())
            num_slices_per_dataset.append(total_slices)
        min_slices = min(num_slices_per_dataset)
        downsampling_factors = [max(1, round(total / min_slices)) for total in num_slices_per_dataset]
        return downsampling_factors

    def _process_splits_common(self, dataset_info: List[Tuple], splits_to_process: List[str], additional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        For each dataset, load metadata, optionally filter subjects, and extract paths
        and slice counts for each requested split (e.g. 'train', 'val', 'test').
        Calls the abstract create_dataset() method (to be implemented in subclasses).
        If 'additional' is provided, passes it to extract_paths_and_count_slices.
        Returns a dictionary keyed by dataset name.
        """
        results = {}
        rank = GPUSetup.get_rank()
        for index, (dataset_name, metadata_path, parquet_folder, mask_labels, instance_bbox, remove_label_ids, downsampling_factor) in enumerate(dataset_info, start=1):
            self.logger.info(f"Rank {rank}: Processing dataset {index}/{len(dataset_info)} with downsampling factor {downsampling_factor}")
            metadata, splits = load_and_process_splits_metadata(metadata_path)
            if self.max_subjects != 'full':
                splits = filter_subjects_by_max_number(splits, max_train_subjects=self.max_subjects)
            dataset_dict = {}
            slice_counts = {}
            subject_ids = {}
            for split in splits_to_process:
                self.logger.info(f"Rank {rank}: Processing {split} split for dataset {dataset_name}")
                if additional is not None:
                    paths, slice_count, subj_ids = extract_paths_and_count_slices(splits[split], parquet_folder, downsampling_factor, additional)
                else:
                    paths, slice_count, subj_ids = extract_paths_and_count_slices(splits[split], parquet_folder, downsampling_factor)

                dataset_inst = self.create_dataset(paths, mask_labels, instance_bbox, remove_label_ids, split)
                dataset_dict[split] = dataset_inst
                slice_counts[f"{split}_slice_count"] = slice_count
                subject_ids[f"{split}_filtered_subjects"] = subj_ids
                # subject_ids[split] = subj_ids
            results[dataset_name] = {
                'datasets': dataset_dict,
                'slice_counts': slice_counts,
                'subject_ids': subject_ids,
                'metadata': metadata,
                'downsampling_factor': downsampling_factor
            }
        return results

    def create_dataset(self, paths: Dict[str, List[str]], mask_labels, instance_bbox, remove_label_ids, split: str):
        """
        Abstract method. Subclasses must override this method to return an instance
        of the appropriate Dataset (e.g. mskSAM2Dataset for finetuning or MultiClassSAM2Dataset for evaluation).
        """
        raise NotImplementedError("Subclasses must implement create_dataset()")

    def collate_fn(self, batch):
        """
        Default collate function to handle variable numbers of boxes.
        Can be overridden if needed.
        """
        images = [item['image'] for item in batch]
        gt2D = [item['gt2D'] for item in batch]
        boxes = [item['boxes'] for item in batch]
        label_ids = [item['label_ids'] for item in batch]
        img_names = [item['img_name'] for item in batch]
        images = torch.stack(images)
        gt2D = torch.stack(gt2D)
        max_num_boxes = max([box.shape[0] for box in boxes])
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
        return {
            'image': images,
            'gt2D': gt2D,
            'boxes': padded_boxes,
            'label_ids': padded_labels,
            'img_name': img_names
        }

    def create_dataloader(self, datasets, batch_size, num_workers, instance_bbox: bool = False):
        """
        Creates DataLoaders from the provided datasets.
          - For finetuning, expects a tuple/list of 3 datasets: (train, val, test).
          - For evaluation, expects a tuple/list of 2 datasets: (sampled, full).
        """
        from torch.utils.data import DataLoader, DistributedSampler
        num_tasks = GPUSetup.get_world_size()
        global_rank = GPUSetup.get_rank()
        
        if isinstance(datasets, (list, tuple)) and len(datasets) == 3:
            train_dataset, val_dataset, test_dataset = datasets
            
            if GPUSetup.is_distributed():
                train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                val_sampler = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                shuffle = False
            else:
                train_sampler = None
                val_sampler = None
                shuffle = False
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=True, sampler=train_sampler, shuffle=shuffle)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                    pin_memory=True, sampler=val_sampler, shuffle=shuffle)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
            
            logger.info(f"Rank {global_rank}: DataLoaders (train, val, test) initialized with batch_size={batch_size}, num_workers={num_workers}")
            return train_loader, val_loader, test_loader
        
        elif isinstance(datasets, (list, tuple)) and len(datasets) == 2:
            
            sampled_test_dataset, full_test_dataset = datasets
            collate_function = self.collate_fn if instance_bbox else None
            
            sampled_test_loader = DataLoader(sampled_test_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, collate_fn=collate_function)
            full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=True, collate_fn=collate_function)
            
            logger.info(f"Rank {global_rank}: DataLoaders (sampled, full) initialized with batch_size={batch_size}, num_workers={num_workers}")
            return sampled_test_loader, full_test_loader
        else:
            raise ValueError("datasets must be a tuple/list of either 2 (evaluation) or 3 (finetuning) dataset objects.")

