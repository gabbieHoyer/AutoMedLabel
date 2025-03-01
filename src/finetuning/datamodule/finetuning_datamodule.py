from typing import Dict, List, Tuple, Any
from torch.utils.data import ConcatDataset

import src.finetuning.utils.gpu_setup as GPUSetup
from .base_datamodule import BaseDataModule
from .components.npy_dataset import mskSAM2Dataset
from .components.experiment_summary import (
    dataset_characteristics,
    add_ml_characteristics,
    aggregate_summaries,
)

class FinetuningDataModule(BaseDataModule):
    """
    DataModule for finetuning.
    Processes three splits: train, val, and test.
    Also aggregates summary statistics (dataset characteristics).
    """
    def __init__(self, dataset_cfg: Dict, augmentation_config: Dict, **kwargs):
        super().__init__(dataset_cfg, **kwargs)
        self.augmentation_config = augmentation_config
        self.splits_to_process = ['train', 'val', 'test']

    def create_dataset(self, paths: Dict[str, List[str]], mask_labels, instance_bbox, remove_label_ids, split: str):
        # Use the finetuning dataset class (mskSAM2Dataset) with split-specific augmentation.
        split_aug_config = self.augmentation_config.get(split, None)
        return mskSAM2Dataset(
            paths['root_paths'],
            paths['gt_paths'],
            paths['img_paths'],
            self.bbox_shift,
            instance_bbox,
            remove_label_ids,
            dataset_name=split,
            augmentation_config=split_aug_config
        )
    
    def process_dataset(self, dataset_info: List[Tuple]) -> Tuple[Any, Any, Any, List[Dict[str, Any]]]:
        # Process splits for each dataset using the shared method.
        results = self._process_splits_common(dataset_info, self.splits_to_process)
        
        # Aggregate datasets for each split.
        aggregated_train = [info['datasets']['train'] for info in results.values()]
        aggregated_val   = [info['datasets']['val'] for info in results.values()]
        aggregated_test  = [info['datasets']['test'] for info in results.values()]
        
        train_dataset = ConcatDataset(aggregated_train) if len(aggregated_train) > 1 else aggregated_train[0]
        val_dataset   = ConcatDataset(aggregated_val)   if len(aggregated_val) > 1 else aggregated_val[0]
        test_dataset  = ConcatDataset(aggregated_test)  if len(aggregated_test) > 1 else aggregated_test[0]
        
        # Compute summaries (only on the main process)
        summaries = []
        dataset_subject_summary = {}
        if GPUSetup.is_main_process():
            for dataset_name, info in results.items():
                ds_summary = dataset_characteristics(info['metadata'], info['slice_counts'], info['subject_ids'])
                ds_summary = add_ml_characteristics(ds_summary, dataset_name, info['downsampling_factor'])
                summaries.append(ds_summary)
                dataset_subject_summary[dataset_name] = info['subject_ids']
            combined_summary = aggregate_summaries(summaries)
            summaries.append(combined_summary)
            if self.max_subjects != 'full':
                summaries.append(dataset_subject_summary)
        return train_dataset, val_dataset, test_dataset, summaries

    def get_loaders(self, batch_size: int, num_workers: int) -> dict:
        dataset_info = self.get_dataset_info()
        train_dataset, val_dataset, test_dataset, summaries = self.process_dataset(dataset_info)
        loaders = self.create_dataloader((train_dataset, val_dataset, test_dataset), batch_size, num_workers, instance_bbox=False)
        return {"loaders": {"train": loaders[0], "val": loaders[1], "test": loaders[2]},
                "summaries": summaries}

