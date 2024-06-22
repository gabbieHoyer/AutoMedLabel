import os
import json
import random
import pandas as pd
from collections import Counter

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
from src.preprocessing.metadata_management.metadata_split import extract_numeric_value

def extract_paths_and_count_slices(subjects, parquet_base_folder, sampling_rate=1, additional_paths=None):
    """Extracts image and mask paths for given subjects from Parquet files, with optional downsampling and additional paths."""
    
    if additional_paths is None:
        additional_paths = []
    
    subject_ids = []
    paths = {'img_paths': [], 'gt_paths': [], 'root_paths': []}
    for path in additional_paths:
        paths[path] = []
    
    total_slices = 0
    sampled_slice_counts = 0
    
    for subject_id in subjects:
        parquet_file = os.path.join(parquet_base_folder, f"{subject_id}.parquet")
        df = pd.read_parquet(parquet_file)
        if sampling_rate > 1:
            # Downsampling: select slices based on the sampling rate
            sampled_df = df.iloc[::sampling_rate, :]
        else:
            sampled_df = df
        paths['img_paths'].extend(sampled_df['npy_image_path'].tolist())
        paths['gt_paths'].extend(sampled_df['npy_mask_path'].tolist())
        paths['root_paths'].extend(sampled_df['npy_base_dir'].tolist())
        for path in additional_paths:
            paths[path].extend(sampled_df[path].tolist())
        total_slices += len(df)
        sampled_slice_counts += len(sampled_df)
        subject_ids.append(subject_id)
    return paths, sampled_slice_counts, subject_ids


def load_and_process_splits_metadata(metadata_file_path):
    """Loads the metadata JSON file and processes it for split information and summary statistics."""
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    # Initialize splits and split_stats
    splits = {'train': [], 'val': [], 'test': []}

    # Process each subject
    for subject_id, details in metadata.items():
        split = details['Split']
        splits[split].append(subject_id)

    return metadata, splits


def filter_subjects_by_max_number(splits, max_train_subjects, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Filter subjects to include only up to the calculated maximum number of subjects for each split.
    
    Parameters:
    - splits: Dictionary of splits with lists of subjects.
    - max_train_subjects: Maximum number of subjects for the training set.
    - train_prop: Proportion of the training set size relative to the total.
    - val_prop: Proportion of the validation set size relative to the total.
    - test_prop: Proportion of the test set size relative to the total.
    """
    # Calculate maximum subjects for validation and test based on proportions

    print('filter subjects!')
    total_subjects = max_train_subjects / train_prop
    max_val_subjects = int(round(total_subjects * val_prop))
    max_test_subjects = int(round(total_subjects * test_prop))

    filtered_splits = {}
    for split_name, subjects in splits.items():
        max_subjects = {
            'train': max_train_subjects,
            'val': max_val_subjects,
            'test': max_test_subjects
        }.get(split_name, len(subjects))  # Default to original size if split name is unexpected

        # Ensure reproducibility by potentially setting a seed before this line if necessary
        random.shuffle(subjects)
        filtered_subjects = subjects[:max_subjects]
        filtered_splits[split_name] = filtered_subjects

    print('return filtered splits!')

    return filtered_splits


def dataset_characteristics(metadata, processed_count_summary, filtered_subject_summary):

    # Initialize splits and split_stats
    splits = {'train': [], 'val': [], 'test': []}
    split_stats = {
        'train': {'num_slices': 0, 'processed_num_slices': 0, 'demographics': {'Sex': Counter(), 'Age': [], 'Weight': []}},
        'val': {'num_slices': 0, 'processed_num_slices': 0, 'demographics': {'Sex': Counter(), 'Age': [], 'Weight': []}},
        'test': {'num_slices': 0, 'processed_num_slices': 0, 'demographics': {'Sex': Counter(), 'Age': [], 'Weight': []}}
    }
    summary = {
        'total_subjects': len(metadata),
        'split_counts': {'train': 0, 'val': 0, 'test': 0},
        'num_slices': 0,
        'processed_num_slices': 0,
        'avg_Age': 0,
        'avg_Weight': 0,
        'sex_counts': Counter()
    }

    # Instead of looping over all subjects in metadata directly,
    # We filter subjects based on those included in filtered_subject_summary
    for split_name in ['train', 'val', 'test']:
        # Get the list of filtered subjects for the current split
        filtered_subjects = set(filtered_subject_summary[f'{split_name}_filtered_subjects'])

        # Process only subjects that are in the filtered list for the current split
        for subject_id in filtered_subjects:
            details = metadata.get(subject_id)
            if not details:
                continue  # Skip if subject ID not found in metadata

            split = details['Split']
            splits[split].append(subject_id)
            summary['split_counts'][split] += 1
            summary['num_slices'] += details['num_slices']
            split_stats[split]['num_slices'] += details['num_slices']
            
            if 'Sex' in details and details['Sex']:
                summary['sex_counts'][details['Sex']] += 1
                split_stats[split]['demographics']['Sex'][details['Sex']] += 1

            if 'Age' in details and details['Age']:
                age = extract_numeric_value(details.get('Age'))
                summary['avg_Age'] += age   #details['Age']
                split_stats[split]['demographics']['Age'].append(age)
           
            if 'Weight' in details and details['Weight']:
                weight = extract_numeric_value(details.get('Weight'))
                summary['avg_Weight'] += weight  #details['Weight']
                split_stats[split]['demographics']['Weight'].append(weight)

    # Calculate overall averages
    summary['avg_Age'] /= summary['total_subjects']
    summary['avg_Weight'] /= summary['total_subjects']

    # Loop over each split to update processed_num_slices, calculate statistics, and clean up
    for split in ['train', 'val', 'test']:
        # Reference to current split's data for easier access
        data = split_stats[split]

        # Update processed_num_slices from processed_count_summary
        data['processed_num_slices'] = processed_count_summary[f'{split}_slice_count']
        # Add to the total processed_num_slices in summary
        summary['processed_num_slices'] += processed_count_summary[f'{split}_slice_count']

        # Calculate demographic statistics if there are subjects in the split
        subject_count = len(splits[split])
        if subject_count > 0:
            data['avg_Age'] = sum(data['demographics']['Age']) / subject_count
            data['avg_Weight'] = sum(data['demographics']['Weight']) / subject_count
            data['sex_counts'] = dict(data['demographics']['Sex'])
        else:
            # Set defaults for empty splits
            data['avg_Age'] = 0
            data['avg_Weight'] = 0
            data['sex_counts'] = {}

        # Remove the detailed demographics to prevent redundancy
        del data['demographics']

    # Convert overall sex counts to a dictionary
    summary['sex_counts'] = dict(summary['sex_counts'])

    # Add split_stats to the summary
    summary['split_stats'] = split_stats

    return summary

def add_ml_characteristics(dataset_summary, dataset_name, downsampling_factor=None):
    modified_summary = dataset_summary.copy()
    modified_summary['dataset_name'] = dataset_name
        
    # Add downsampling factor to the summary if applicable
    if downsampling_factor is not None:
        modified_summary['downsampling_factor'] = downsampling_factor
    else:
        # In case downsampling_factors were not provided or are not applicable
        modified_summary['downsampling_factor'] = None
    
    return modified_summary

def aggregate_summaries(summaries):
    """Aggregate summaries from multiple datasets, including correct split-level statistics."""
    combined_summary = {
        'total_subjects': 0,
        'split_counts': {'train': 0, 'val': 0, 'test': 0},
        'num_slices': 0,
        'processed_num_slices': 0,
        'avg_Age': 0,
        'avg_Weight': 0,
        'sex_counts': Counter(),
        'split_stats': {
            'train': {'num_slices': 0, 'processed_num_slices': 0, 'avg_Age': 0, 'avg_Weight': 0, 'sex_counts': Counter()},
            'val': {'num_slices': 0, 'processed_num_slices': 0, 'avg_Age': 0, 'avg_Weight': 0, 'sex_counts': Counter()},
            'test': {'num_slices': 0, 'processed_num_slices': 0, 'avg_Age': 0, 'avg_Weight': 0, 'sex_counts': Counter()}
        }
    }

    split_subject_counts = {'train': 0, 'val': 0, 'test': 0}

    # Initialize total age and weight for weighted averages calculation
    total_weighted_age = 0
    total_weighted_weight = 0

    for summary in summaries:
        combined_summary['total_subjects'] += summary['total_subjects']
        combined_summary['num_slices'] += summary['num_slices']
        combined_summary['processed_num_slices'] += summary['processed_num_slices']
        combined_summary['sex_counts'] += Counter(summary['sex_counts'])

        # Accumulate weighted ages and weights
        total_weighted_age += summary['avg_Age'] * summary['total_subjects']
        total_weighted_weight += summary['avg_Weight'] * summary['total_subjects']

        for split in ['train', 'val', 'test']:
            split_summary = summary['split_stats'][split]
            combined_summary['split_counts'][split] += sum(split_summary['sex_counts'].values())
            combined_summary['split_stats'][split]['num_slices'] += split_summary['num_slices']
            combined_summary['split_stats'][split]['processed_num_slices'] += split_summary['processed_num_slices']
            combined_summary['split_stats'][split]['sex_counts'] += Counter(split_summary['sex_counts'])
            
            # Accumulate for weighted average calculation
            combined_summary['split_stats'][split]['avg_Age'] += split_summary['avg_Age'] * sum(split_summary['sex_counts'].values())
            combined_summary['split_stats'][split]['avg_Weight'] += split_summary['avg_Weight'] * sum(split_summary['sex_counts'].values())
            split_subject_counts[split] += sum(split_summary['sex_counts'].values())

    # Compute overall averages for age and weight
    if combined_summary['total_subjects'] > 0:
        combined_summary['avg_Age'] = total_weighted_age / combined_summary['total_subjects']
        combined_summary['avg_Weight'] = total_weighted_weight / combined_summary['total_subjects']

    # Finalize split-level averages
    for split in ['train', 'val', 'test']:
        if split_subject_counts[split] > 0:
            combined_summary['split_stats'][split]['avg_Age'] /= split_subject_counts[split]
            combined_summary['split_stats'][split]['avg_Weight'] /= split_subject_counts[split]

        combined_summary['split_stats'][split]['sex_counts'] = dict(combined_summary['split_stats'][split]['sex_counts'])

    combined_summary['sex_counts'] = dict(combined_summary['sex_counts'])
    combined_summary['dataset_name'] = 'wholistic_dataset'
    combined_summary['dataset_name'] = 'combined'
    combined_summary['downsampling_factor'] = None  # No downsampling factor for combined
    
    return combined_summary

    


# def extract_paths_and_count_slices(subjects, parquet_base_folder, sampling_rate=1):
#     """Extracts image and mask paths for given subjects from Parquet files, with optional downsampling."""
    
#     subject_ids = []
#     paths = {'img_paths': [], 'gt_paths': [], 'root_paths': []}
#     total_slices = 0
#     sampled_slice_counts = 0
    
#     for subject_id in subjects:
#         parquet_file = os.path.join(parquet_base_folder, f"{subject_id}.parquet")
#         df = pd.read_parquet(parquet_file)
#         if sampling_rate > 1:
#             # Downsampling: select slices based on the sampling rate
#             sampled_df = df.iloc[::sampling_rate, :]
#         else:
#             sampled_df = df
#         paths['img_paths'].extend(sampled_df['npy_image_path'].tolist())
#         paths['gt_paths'].extend(sampled_df['npy_mask_path'].tolist())
#         paths['root_paths'].extend(sampled_df['npy_base_dir'].tolist())
#         total_slices += len(df)
#         sampled_slice_counts += len(sampled_df)
#         subject_ids.append(subject_id)
#     return paths, sampled_slice_counts, subject_ids


# def filter_subjects_by_max_number(splits, max_subjects):
#     """Filter subjects to include only up to max_subjects from each split."""
#     filtered_splits = {}
#     for split_name, subjects in splits.items():
#         # Sort subjects based on some criteria, e.g., randomly or based on their metadata like 'Age' or 'Sex'
#         # Here, we'll do it randomly. Ensure reproducibility by setting a seed if necessary.
#         random.shuffle(subjects)
#         filtered_subjects = subjects[:max_subjects]
#         filtered_splits[split_name] = filtered_subjects
#     return filtered_splits



# def summarize_and_save(experiment_config, summaries, downsampling_factors=None):
#     # Check if summaries is a dictionary (single dataset case), wrap it in a list
#     if isinstance(summaries, dict):
#         summaries = [summaries]
    
#     # Initialize an empty list to hold modified summaries with dataset names
#     modified_summaries = []

#     # Iterate through summaries and add dataset names
#     for dataset_name, summary in zip(experiment_config['datasets'].keys(), summaries):
#         # Copy the summary to avoid modifying the original summary in place
#         modified_summary = summary.copy()
#         modified_summary['dataset_name'] = dataset_name
#         modified_summaries.append(modified_summary)

#     # If there is more than one dataset, add the combined summary
#     if len(modified_summaries) > 1:
#         combined_summary = aggregate_summaries(modified_summaries)
#         combined_summary['dataset_name'] = 'combined'
#         modified_summaries.append(combined_summary)

#     # Convert the list of modified summaries to a DataFrame
#     summary_df = pd.DataFrame(modified_summaries)
    
#     # Save the DataFrame to CSV
#     summary_file_path = os.path.join(experiment_config['output_configuration']['save_path'], experiment_config['output_configuration']['summary_file'])
#     summary_df.to_csv(summary_file_path, index=False)

#     print(f"Summary saved to {summary_file_path}")