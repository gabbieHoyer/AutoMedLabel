import os
import copy

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

# --------------- Main class to load and manipulate config files ---------------
from src.utils.file_management.config_loader import ConfigPathManager
from src.utils.file_management.args_handler import apply_overrides, apply_overrides2

# -------- Tool blocks to load and combine config files for experimental use cases --------
def load_and_merge_config_section(main_config, section_path, config_path_manager, sub_dir, fields_to_merge='all'):
    """
    Loads and merges a config section based on a path within the main config.
    """
    config_name = main_config
    for key in section_path[:-1]:
        config_name = config_name.get(key, {})
    
    # Extract the config file name for the final key in the path
    config_name = config_name.get(section_path[-1], {}).get('config', None)

    if isinstance(config_name, str):
        additional_config_path = config_path_manager.extract_config_path(file_name=config_name, sub_dir=sub_dir)
        additional_config = config_path_manager.load_config_yaml_path(additional_config_path)
        
        # Merge the additional config with the main config
        main_config = config_path_manager.merge_missing_config_values(main_config, additional_config, fields_to_merge)  
    else:
        print(f"No valid 'config' found for section {'/'.join(section_path)}. Proceeding without it.")
    return main_config


def load_and_merge_visualization_configs(main_config, config_path_manager, sub_dir):
    """
    Specifically handles loading and merging visualization configurations.
    Assumes each visualization config to be merged under its respective key within the `visualizations` section.
    """
    visualization_configs = main_config.get('visualizations', {})
    for key, value in visualization_configs.items():
        config_name = value.get('config', None)
        if isinstance(config_name, str):
            # Construct the full path and check if it exists
            additional_config_path = os.path.join(config_path_manager.base_dir, sub_dir, config_name)
            
            if os.path.exists(additional_config_path):
                additional_config_path = config_path_manager.extract_config_path(file_name=config_name, sub_dir=sub_dir)
                additional_config = config_path_manager.load_config_yaml_path(additional_config_path)
                # Merge this config back under its unique key within `visualizations`
                # Assuming each config file starts with a `visualizations` key
                if 'visualizations' in additional_config:
                    main_config['visualizations'][key] = additional_config['visualizations']
                else:
                    print(f"Warning: No 'visualizations' key found in {config_name}. Skipping.")
            else:
                print(f"Warning: Visualization config file {additional_config_path} not found. Skipping.")

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

# ------------ Primary Pipeline Functions to load and manipulate config files ------------
def load_dataset_config(config_file_name, base_dir):
    configPathManager = ConfigPathManager(base_dir=base_dir)
    #Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=os.path.join('preprocessing','datasets'))
    #Load config file
    main_config = configPathManager.load_config_yaml_path(config_path)
    return main_config

# ------------------------------------------------------------------

def load_autolabel_prompt(config_file_name, base_dir):
    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=os.path.join('prompting','autolabeling'))
    main_config = configPathManager.load_config_yaml_path(config_path)

    # # Process each dataset configuration
    for dataset_name in main_config.get('dataset', {}).keys():

        new_fields_to_add = [
            ['data', 'mask_labels'],  # only works if the field is empty in my current config
            ['preprocessing_cfg', 'image_size'],
            ['preprocessing_cfg', 'voxel_num_thre2d'],
            ['preprocessing_cfg', 'voxel_num_thre3d'],
            ['preprocessing_cfg', 'instance_bbox'],
            ['data', 'ml_metadata_file'],
            ['data', 'slice_info_parquet_dir'],
        ]
        default_fields_to_add = [
            ['mask_labels'],
            ['preprocessing_cfg', 'image_size'],
            ['preprocessing_cfg', 'voxel_num_thre2d'],
            ['preprocessing_cfg', 'voxel_num_thre3d'],
            ['preprocessing_cfg', 'instance_bbox'],
            ['ml_metadata_file'],
            ['slice_info_parquet_dir'],
        ]

        main_config = load_and_merge_dataset_config(main_config, dataset_name, configPathManager, 'preprocessing/datasets', new_fields_to_add, default_fields_to_add)

    return main_config

# ------------------------------------------------------------------

def load_experiment(config_file_name, base_dir, kwargs=None):
    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=os.path.join('finetuning','experiments'))
    main_config = configPathManager.load_config_yaml_path(config_path)

    # # Apply terminal argument updates
    if kwargs: #!=None:
        apply_overrides(main_config, kwargs)

    # Update config with augmentation_pipeline info using the refactored function
    main_config = load_and_merge_config_section(main_config, ['datamodule', 'augmentation_pipeline'], configPathManager, 'preprocessing/augmentations')

    # Handle visualization configs separately
    main_config = load_and_merge_visualization_configs(main_config, configPathManager, 'finetuning/evaluation')

    # Process each dataset configuration
    for dataset_name in main_config.get('dataset', {}).keys():

        new_fields_to_add = [
            ['dataset', dataset_name, 'name'],
            ['dataset', dataset_name, 'ml_metadata_file'],
            ['dataset', dataset_name, 'slice_info_parquet_dir'],
            ['dataset', dataset_name, 'mask_labels'],
            ['dataset', dataset_name, 'instance_bbox'],
            ['dataset', dataset_name, 'remove_label_ids'],
        ]
        default_fields_to_add = [
            ['dataset', 'name'],
            ['ml_metadata_file'],
            ['slice_info_parquet_dir'],
            ['mask_labels'],
            ['preprocessing_cfg', 'instance_bbox'],
            ['preprocessing_cfg', 'remove_label_ids']
        ]
        # ['prompt_experiment', 'preprocessing_cfg', 'instance_bbox'],

        main_config = load_and_merge_dataset_config(main_config, dataset_name, configPathManager, 'preprocessing/datasets', new_fields_to_add, default_fields_to_add)

    # # Apply terminal argument updates
    # if kwargs !=None:
    #     apply_overrides2(main_config, kwargs, dataset)

    return main_config


def load_evaluation(config_file_name, base_dir, kwargs=None):

    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=os.path.join('finetuning','evaluation'))
    main_config = configPathManager.load_config_yaml_path(config_path)

    # Apply terminal argument updates
    if kwargs !=None:
        apply_overrides(main_config, kwargs)
    
    # import pdb; pdb.set_trace()

    # Process each dataset configuration
    for dataset_name in main_config.get('dataset', {}).keys():

        new_fields_to_add = [
            ['dataset', dataset_name, 'name'],
            ['dataset', dataset_name, 'ml_metadata_file'],
            ['dataset', dataset_name, 'stats_metadata_file'],
            ['dataset', dataset_name, 'slice_info_parquet_dir'],
            ['dataset', dataset_name, 'mask_labels'],
            ['dataset', dataset_name, 'instance_bbox'],
            ['dataset', dataset_name, 'remove_label_ids'],
        ]
        default_fields_to_add = [
            ['dataset', 'name'],
            ['ml_metadata_file'],
            ['stats_metadata_file'],
            ['slice_info_parquet_dir'],
            ['mask_labels'],
            ['preprocessing_cfg', 'instance_bbox'],
            ['preprocessing_cfg', 'remove_label_ids']
        ]
        # import pdb; pdb.set_trace()
        main_config = load_and_merge_dataset_config(main_config, dataset_name, configPathManager, 'preprocessing/datasets', new_fields_to_add, default_fields_to_add)

    return main_config

    
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

