# src/utils/config_handler.py
import os
import os.path as path
from typing import Any, Dict, List, Union

# --------------- Main class to load and manipulate config files ---------------
from .config_loader import ConfigPathManager, load_yaml
from .args_handler import apply_overrides, apply_overrides2

# -------- Tool blocks to load and combine config files for experimental use cases --------

def load_and_merge_config_section(main_config: dict, section_path: List[str],
                                  manager: ConfigPathManager, sub_dir: str,
                                  fields_to_merge: Union[str, List[List[str]]] = 'all') -> dict:
    """
    Look up a nested section in main_config, load an additional YAML file specified in that section,
    and merge missing values from it into main_config.
    """
    section = main_config
    for key in section_path[:-1]:
        section = section.get(key, {})
    config_file = section.get(section_path[-1], {}).get('config')
    if config_file and isinstance(config_file, str):
        add_path = manager.get_config_path(config_file, sub_dir)
        additional_config = manager.load_config_yaml(add_path)
        main_config = manager.merge_missing_values(main_config, additional_config, fields_to_merge)
    else:
        print(f"No valid config found for section {'/'.join(section_path)}; skipping merge.")
    return main_config

def load_and_merge_dataset_config(main_config, dataset_name, config_path_manager, sub_dir, new_fields_to_add, default_fields_to_add):
    """
    Loads a dataset config and merges it into the main config with specified fields.
    """
    # Extract path to config and make sure yaml file exists
    additional_config_path = config_path_manager.extract_config_path(file_name=main_config.get('dataset', {}).get(dataset_name, {}).get('config'), 
                                                                     sub_dir=sub_dir)
    # Load config file
    dataset_config = config_path_manager.load_config_yaml_path(additional_config_path)

    # Update missing fields in main config file using corresponding fields from the additional config file
    main_config = config_path_manager.add_new_config_values(main_config, dataset_config, new_fields_to_add, default_fields_to_add)

    return main_config

def load_dataset_config(config_file_name: str, base_dir: str) -> dict:
    manager = ConfigPathManager(base_dir)
    config_path = manager.get_config_path(config_file_name, sub_dir=os.path.join('preprocessing', 'datasets'))
    return manager.load_config_yaml(config_path)

def load_experiment(config_file_name: str, base_dir: str, kwargs: dict = None) -> dict:
    manager = ConfigPathManager(base_dir)
    # 1) Load the main experiment config
    config_path = manager.get_config_path(config_file_name, sub_dir=os.path.join('finetuning', 'experiments'))
    main_config = manager.load_config_yaml(config_path)

    if kwargs:
        apply_overrides(main_config, kwargs)

    # 2) Merge augmentation pipeline config, if needed
    main_config = load_and_merge_config_section(
        main_config,
        ['datamodule', 'augmentation_pipeline'],
        manager,
        sub_dir='preprocessing/augmentations'
    )

    # 3) For each dataset, load its config from “preprocessing/datasets” and merge new fields
    # inside load_experiment or load_evaluation:
    for dataset_name in main_config.get('dataset', {}):
        dataset_cfg_file = main_config['dataset'][dataset_name].get('config')
        if not dataset_cfg_file:
            continue  # skip or raise error

        path = manager.get_config_path(dataset_cfg_file, sub_dir='preprocessing/datasets')
        dataset_cfg = manager.load_config_yaml(path)

        # The pairs of (new_key_path, default_key_path):
        new_fields = [
            ['dataset', dataset_name, 'name'],
            ['dataset', dataset_name, 'ml_metadata_file'],
            ['dataset', dataset_name, 'slice_info_parquet_dir'],
            ['dataset', dataset_name, 'mask_labels'],
            ['dataset', dataset_name, 'instance_bbox'],
            ['dataset', dataset_name, 'remove_label_ids'],
        ]
        default_fields = [
            ['dataset', 'name'], 
            ['ml_metadata_file'],
            ['slice_info_parquet_dir'],
            ['mask_labels'],
            ['preprocessing_cfg','instance_bbox'],
            ['preprocessing_cfg','remove_label_ids'],
        ]

        # Then call your improved function that uses distinct paths:
        main_config = manager.add_new_values(main_config, dataset_cfg, new_fields, default_fields)
    
    return main_config

def load_evaluation(
    config_file_name: str,
    base_dir: str,
    kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Load and merge an evaluation configuration for finetuning evaluation.
    This function is similar to load_experiment() but looks in the
    finetuning/evaluation subfolder for the main config, then merges each dataset
    config from 'preprocessing/datasets'.
    """
    manager = ConfigPathManager(base_dir)
    # 1) Load the main evaluation config
    config_path = manager.get_config_path(
        config_file_name,
        sub_dir=os.path.join('finetuning', 'evaluation')
    )
    main_config = manager.load_config_yaml(config_path)

    # 2) Optionally apply command-line overrides
    if kwargs:
        apply_overrides(main_config, kwargs)

    # 3) For each dataset, load its config from “preprocessing/datasets” and merge new fields
    for dataset_name in main_config.get('dataset', {}):
        dataset_cfg_file = main_config['dataset'][dataset_name].get('config')
        if not dataset_cfg_file:
            continue  # skip or raise error if needed

        # Load the dataset YAML
        path = manager.get_config_path(dataset_cfg_file, sub_dir='preprocessing/datasets')
        dataset_cfg = manager.load_config_yaml(path)

        # The pairs of (new_key_path, default_key_path):
        new_fields = [
            ['dataset', dataset_name, 'name'],
            ['dataset', dataset_name, 'ml_metadata_file'],
            ['dataset', dataset_name, 'stats_metadata_file'],
            ['dataset', dataset_name, 'slice_info_parquet_dir'],
            ['dataset', dataset_name, 'mask_labels'],
            ['dataset', dataset_name, 'instance_bbox'],
            ['dataset', dataset_name, 'remove_label_ids'],
            ['dataset', dataset_name, 'voxel_num_thre2d'],
            ['dataset', dataset_name, 'kernel_size'],
        ]
        default_fields = [
            ['dataset', 'name'],
            ['ml_metadata_file'],
            ['stats_metadata_file'],
            ['slice_info_parquet_dir'],
            ['mask_labels'],
            ['preprocessing_cfg', 'instance_bbox'],
            ['preprocessing_cfg', 'remove_label_ids'],
            ['preprocessing_cfg', 'voxel_num_thre2d'],
            ['preprocessing_cfg', 'kernel_size'],
        ]

        # 4) Copy from `dataset_cfg` into `main_config` for the missing keys
        main_config = manager.add_new_values(main_config, dataset_cfg, new_fields, default_fields)

    return main_config

# --------------------------------------------------------- #

def load_autolabel_config(config_file_name: str, base_dir: str) -> dict:
    """
    Loads an autolabel configuration file from the 'config/obj_detection/inference' directory.
    """
    manager = ConfigPathManager(base_dir)
    config_path = manager.get_config_path(config_file_name, sub_dir=path.join("obj_detection", "inference"))
    return manager.load_config_yaml(config_path)

def load_det_config(config_file_name: str, base_dir: str) -> dict:
    """
    Loads a detection experiment configuration file from the 'config/obj_detection/experiments' directory.
    """
    manager = ConfigPathManager(base_dir)
    config_path = manager.get_config_path(config_file_name, sub_dir=path.join("obj_detection", "experiments"))
    return manager.load_config_yaml(config_path)
    
# ---------------- SUMMARIZE CONFIG FILE ------------------ #
def summarize_config(config, path):
    def _summarize(current_config, indent_level=0):
        summary_lines = []
        indent_space = '  ' * indent_level  # Two spaces per indent level

        for key, value in current_config.items():
            if isinstance(value, dict):
                summary_lines.append(f"{indent_space}{key}:")
                summary_lines.extend(_summarize(value, indent_level + 1))
            else:
                summary_lines.append(f"{indent_space}{key}: {value}")

        return summary_lines
    
    # Check if the path exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)  # This will create all intermediate-level directories needed to contain the leaf directory

    summary_lines = _summarize(config)
    summary_text = '\n'.join(summary_lines)

    with open(f"{path}/config_file_summary.txt", "w") as summary_file:
        summary_file.write(summary_text)

