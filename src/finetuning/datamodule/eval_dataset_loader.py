""" 
Script in Development for 3D dice computation
Finetuning Evaluation Script for 2D, 
class-based dice computation on Test sets
"""


import os
import json
import random
import logging
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

import torch

from torch.utils.data import DataLoader 
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import src.finetuning.utils.gpu_setup as GPUSetup
from src.utils.file_management.config_loader import load_yaml
from src.finetuning.datamodule.experiment_summary import (
    load_and_process_splits_metadata, 
    extract_paths_and_count_slices,
    filter_subjects_by_max_number,
)
from src.finetuning.datamodule.npy_dataset import MultiClassSAMDataset

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def get_dataset_info(dataset_cfg, is_balanced:bool=False):
    """
    Extract information to create dataloaders.
    """
    def calculate_downsampling_factors(metadata_paths):
        """Calculate downsampling factors based on num_slices values loaded from YAML configuration files, accumulated for each subject."""
        num_slices_per_dataset = []
        
        for path in metadata_paths:
            config = load_yaml(path)

            # Accumulate num_slices for each subject in the dataset
            total_slices = sum(subject['num_slices'] for subject in config.values())

            if GPUSetup.is_main_process():
                print('total slices!', total_slices) 
            num_slices_per_dataset.append(total_slices)

        if GPUSetup.is_main_process():
            print('num slices per dataset!', num_slices_per_dataset)
        # Find the dataset with the minimum number of slices to base downsampling factors on
        min_slices = min(num_slices_per_dataset)

        if GPUSetup.is_main_process():
            print('min slices!', min_slices) # for testing only - will remove
        
        # Calculate downsampling factors, ensuring at least every 2nd slice is selected
        downsampling_factors = [max(1, round(total_slices / min(num_slices_per_dataset))) for total_slices in num_slices_per_dataset]
        
        return downsampling_factors

    rank = GPUSetup.get_rank()
    logger.info(f"Rank {rank}: Starting to prep datasets")
    # Extract data from cfg
    dataset_name_list = []
    ml_metadata_file_list = []
    slice_info_parquet_dir_list = []
    mask_labels_list = []
    instance_bbox_list = []
    remove_label_ids_list = []
    for dataset_name in dataset_cfg.keys():
        dataset_name_list.append(dataset_name)
        mask_labels_list.append(dataset_cfg.get(dataset_name).get('mask_labels'))
        instance_bbox_list.append(dataset_cfg.get(dataset_name).get('instance_bbox'))
        remove_label_ids_list.append(dataset_cfg.get(dataset_name).get('remove_label_ids'))
        ml_metadata_file_list.append(dataset_cfg.get(dataset_name).get('ml_metadata_file'))
        slice_info_parquet_dir_list.append(dataset_cfg.get(dataset_name).get('slice_info_parquet_dir'))
    
    # Check if balanced loading is enabled
    if is_balanced == True:
        logger.info(f"Rank {rank}: Calculating downsampling factors for balanced loading")
        downsampling_factors = calculate_downsampling_factors(ml_metadata_file_list)
    else:
        logger.info(f"Rank {rank}: Balanced loading not enabled, proceeding without downsampling")
        downsampling_factors = [1] * len(dataset_name_list)

    if GPUSetup.is_main_process():
        print('downsampling_factors:')
        print(downsampling_factors)   # for testing only - will remove

    dataset_info = list(zip(dataset_name_list, ml_metadata_file_list, slice_info_parquet_dir_list, mask_labels_list, instance_bbox_list, remove_label_ids_list, downsampling_factors))
    logger.info(f"Rank {rank}: Starting to prep datasets")

    return dataset_info

def process_dataset(dataset_info:list[tuple], bbox_shift:int, max_subjects:Optional[int | str] = 'full'):

    rank = GPUSetup.get_rank()
    logger.info(f"Rank {rank}: Starting to process datasets")

    # Initialize lists to hold aggregated datasets
    aggregated_sampled_test_datasets = []
    aggregated_full_test_datasets = []

    # Iterate through each dataset
    for index, (dataset_name, metadata_path, parquet_folder, mask_labels, instance_bbox, remove_label_ids, downsampling_factor) in enumerate(dataset_info, start=1):
        logger.info(f"Rank {rank}: Processing dataset {index}/{len(dataset_info)} with downsampling factor {downsampling_factor}")
        
        # Load metadata and all splits
        metadata, full_splits = load_and_process_splits_metadata(metadata_path)

        # If max_subjects is an integer, filter to get a sampled version of the splits
        sampled_splits = filter_subjects_by_max_number(full_splits, max_train_subjects=max_subjects) if isinstance(max_subjects, int) else None

        # Define function to process and create dataset from splits
        def create_datasets_from_splits(splits, is_sampled):
            for split_name in ['train', 'val', 'test']:
                logger.info(f"Rank {rank}: Processing {split_name} split")
                paths, slice_count, subject_ids = extract_paths_and_count_slices(splits[split_name], parquet_folder, downsampling_factor)
                
                dataset = MultiClassSAMDataset(
                    paths['root_paths'],
                    paths['gt_paths'],
                    paths['img_paths'],
                    bbox_shift,
                    mask_labels,
                    instance_bbox,
                    remove_label_ids,
                )
                
                if split_name == 'test':
                    if is_sampled:
                        aggregated_sampled_test_datasets.append(dataset)
                    else:
                        aggregated_full_test_datasets.append(dataset)

        # Always create dataset from full splits
        create_datasets_from_splits(full_splits, False)

        # Optionally create dataset from sampled splits if max_subjects is an integer
        if isinstance(max_subjects, int):
            create_datasets_from_splits(sampled_splits, True)

    # Combine datasets if multiple datasets are processed or return the first element if only one
    full_test_dataset = ConcatDataset(aggregated_full_test_datasets) if len(aggregated_full_test_datasets) > 1 else aggregated_full_test_datasets[0]
    sampled_test_dataset = ConcatDataset(aggregated_sampled_test_datasets) if len(aggregated_sampled_test_datasets) > 1 else (aggregated_sampled_test_datasets[0] if aggregated_sampled_test_datasets else None)
          
    logger.info(f"Rank {rank}: Finished processing datasets")
    return sampled_test_dataset, full_test_dataset


def collate_fn(batch):
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

    
def create_dataloader(datasets, batch_size, num_workers, instance_bbox):
    sampled_test_dataset, full_test_dataset = datasets

    num_tasks = GPUSetup.get_world_size()
    global_rank = GPUSetup.get_rank()

    if instance_bbox:
        collate_function = collate_fn
    else:
        collate_function = None

    sampled_test_loader  = DataLoader(sampled_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_function)
    full_test_loader  = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_function)
    
    # After DataLoader initialization
    logger.info(f"Rank {global_rank}: DataLoaders initialized with batch_size={batch_size}, num_workers={num_workers}")

    return sampled_test_loader, full_test_loader







                # if instance_bbox:
                #     dataset = MultiInstanceClassNpyDataset(
                #         paths['root_paths'],
                #         paths['gt_paths'],
                #         paths['img_paths'],
                #         bbox_shift,
                #         mask_labels,
                #         instance_bbox,
                #         remove_label_ids,
                #     )

                # else:
                #     dataset = MultiClassNpyDataset(
                #         paths['root_paths'],
                #         paths['gt_paths'],
                #         paths['img_paths'],
                #         bbox_shift,
                #         mask_labels,
                #         instance_bbox,
                # )