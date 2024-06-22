""" Finetuning scripts functional for non instance-based finetuning """

import os
import json
import random
import logging
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

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
    dataset_characteristics,
    add_ml_characteristics,
    aggregate_summaries,
)
# from src.finetuning.datamodule.npy_dataset import NpyDataset_og, NpyRandInstanceGTMatchedDataset
# NpyRandInstanceGTMatchedDataset if instance
# NpyDataset_og if not instance
# Eventually make this a single dataset

from src.finetuning.datamodule.npy_dataset import mskSAMDataset

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

def process_dataset(dataset_info:list[tuple], augmentation_config:dict, bbox_shift:int, max_subjects:Optional[int | str] = 'full'):

    rank = GPUSetup.get_rank()
    logger.info(f"Rank {rank}: Starting to process datasets")

    # Initialize lists to hold aggregated datasets
    aggregated_train_datasets = []
    aggregated_val_datasets = []
    aggregated_test_datasets = []
    # Initialize lists to hold aggregated dataset summary
    summaries = [] 
    dataset_subject_summary = {}
    # Iterate through each dataset
    for index, (dataset_name, metadata_path, parquet_folder, mask_labels, instance_bbox, remove_label_ids, downsampling_factor) in enumerate(dataset_info, start=1):
        logger.info(f"Rank {rank}: Processing dataset {index}/{len(dataset_info)} with downsampling factor {downsampling_factor}")
        
        count_summary = {'train_slice_count': 0, 'val_slice_count': 0, 'test_slice_count': 0}
        subject_summary = {'train_filtered_subjects': [], 'val_filtered_subjects': [], 'test_filtered_subjects': []}
        
        metadata, splits = load_and_process_splits_metadata(metadata_path)

        # subject number selection functionality:
        if max_subjects != 'full':
            splits = filter_subjects_by_max_number(splits, max_train_subjects=max_subjects)
        
        # Process each split and count slices
        for split_name in ['train', 'val', 'test']:
            logger.info(f"Rank {rank}: Processing {split_name} split")
            logger.info("Extracts image and mask paths for given subjects from Parquet files, with optional downsampling.")
            paths, slice_count, subject_ids = extract_paths_and_count_slices(splits[split_name], parquet_folder, downsampling_factor)

            # Note: Assuming the same augmentation pipeline for 'val' and 'test'
            split_augmentation_config = augmentation_config.get(split_name, None)

            dataset = mskSAMDataset(
                paths['root_paths'],
                paths['gt_paths'],
                paths['img_paths'],
                bbox_shift,
                instance_bbox,
                remove_label_ids,
                dataset_name,
                augmentation_config=split_augmentation_config
            )
            # in future, could pass mask_labels if attempting inline metric calc

            # Aggregate datasets and update summary accordingly
            if split_name == 'train':
                aggregated_train_datasets.append(dataset)
                count_summary['train_slice_count'] += slice_count
                subject_summary['train_filtered_subjects'].extend(subject_ids)
            elif split_name == 'val':
                aggregated_val_datasets.append(dataset)
                count_summary['val_slice_count'] += slice_count
                subject_summary['val_filtered_subjects'].extend(subject_ids)
            else:  # 'test'
                aggregated_test_datasets.append(dataset)
                count_summary['test_slice_count'] += slice_count
                subject_summary['test_filtered_subjects'].extend(subject_ids)

        # Perform summary computation only on rank 0
        if GPUSetup.is_main_process():
            dataset_summary = dataset_characteristics(metadata, count_summary, subject_summary)
            dataset_summary = add_ml_characteristics(dataset_summary, dataset_name, downsampling_factor)
            summaries.append(dataset_summary)
            dataset_subject_summary[dataset_name] = subject_summary

    if len(dataset_info) > 1:
        # Combine the datasets for each phase using your preferred method (e.g., ConcatDataset)
        train_dataset = ConcatDataset(aggregated_train_datasets)
        val_dataset = ConcatDataset(aggregated_val_datasets)
        test_dataset = ConcatDataset(aggregated_test_datasets)

    else:
        train_dataset = aggregated_train_datasets[0]
        val_dataset = aggregated_val_datasets[0]
        test_dataset = aggregated_test_datasets[0]

    # Perform summary computation only on rank 0
    if GPUSetup.is_main_process():
        combined_summary = aggregate_summaries(summaries)  
        summaries.append(combined_summary)
        
        if max_subjects != 'full':
            summaries.append(dataset_subject_summary)
          
    logger.info(f"Rank {rank}: Finished processing datasets")
    return train_dataset, val_dataset, test_dataset, summaries

def save_dataset_summary(summaries:List[Dict[str,Any]], summary_file_path:str, max_subjects:Optional[int | str] = 'full'):
    # Update and save summary for all datasets only on rank 0
    if GPUSetup.is_main_process():
        logger.info(f"Rank {GPUSetup.get_rank()}: Summarizing and saving dataset characteristics")
        
        # Extract the directory path from summary_file_path
        directory_path = os.path.dirname(summary_file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if max_subjects != 'full':
            # Assume the last element in summaries contains the subject_ids information for multiple datasets
            dataset_subject_summary = summaries.pop()  # Remove and save the last element

            # Prepare the path for the JSON file
            subject_ids_file_path = os.path.join(directory_path, "filtered_subject_ids.json")

            with open(subject_ids_file_path, 'w') as f:
                clean_dataset_subject_summary = {}
                for dataset_name, subject_ids_info in dataset_subject_summary.items():
                    # For each dataset, clean the keys from '..._filtered_subjects' to 'train', 'val', 'test'
                    clean_subject_ids_info = {
                        split.replace('_filtered_subjects', ''): ids
                        for split, ids in subject_ids_info.items()
                    }
                    clean_dataset_subject_summary[dataset_name] = clean_subject_ids_info
                
                # Dump the cleaned and structured subject IDs info for all datasets into the JSON file
                json.dump(clean_dataset_subject_summary, f, indent=4)

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(summary_file_path, index=False, mode='w')  # will overwrite if file already exists in provided location
        
        del summary_df  # Delete the dataframe explicitly
        import gc
        gc.collect()    # Force garbage collection

    return True
    
def create_dataloader(datasets, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = datasets

    num_tasks = GPUSetup.get_world_size()
    global_rank = GPUSetup.get_rank()
    
    # Use DistributedSampler for the training/val datasets in a distributed setup
    if GPUSetup.is_distributed():
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        shuffle = False  # Shuffle is handled by the DistributedSampler
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=train_sampler, shuffle=shuffle)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=val_sampler, shuffle=shuffle)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # After DataLoader initialization
    logger.info(f"Rank {global_rank}: DataLoaders initialized with batch_size={batch_size}, num_workers={num_workers}")

    return train_loader, val_loader, test_loader









            # if instance_bbox:
            #     dataset = NpyRandInstanceGTMatchedDataset(
            #         paths['root_paths'],
            #         paths['gt_paths'],
            #         paths['img_paths'],
            #         bbox_shift,
            #         instance_bbox,
            #         remove_label_ids,
            #         augmentation_config=split_augmentation_config
            #     )
            # else:
            #     dataset = NpyDataset_og(
            #         paths['root_paths'],
            #         paths['gt_paths'],
            #         paths['img_paths'],
            #         bbox_shift,
            #         remove_label_ids,
            #         augmentation_config=split_augmentation_config
            #     )