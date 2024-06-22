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
    for key in section_path[:-1]:  # Traverse to the last key
        config_name = config_name.get(key, {})
    config_name = config_name.get(section_path[-1], None)  # Get config file name

    if isinstance(config_name, str):
        additional_config_path = config_path_manager.extract_config_path(file_name=config_name, sub_dir=sub_dir)
        additional_config = config_path_manager.load_config_yaml_path(additional_config_path)
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
            additional_config_path = config_path_manager.extract_config_path(file_name=config_name, sub_dir=sub_dir)
            additional_config = config_path_manager.load_config_yaml_path(additional_config_path)
            # Merge this config back under its unique key within `visualizations`
            # Assuming each config file starts with a `visualizations` key
            if 'visualizations' in additional_config:
                main_config['visualizations'][key] = additional_config['visualizations']
            else:
                print(f"Warning: No 'visualizations' key found in {config_name}. Skipping.")

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


def load_prompting_experiment(config_file_name, base_dir, prompt_type='zeroshot'):
    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Conditionally set base_config_path based on prompt_type
    if prompt_type == 'zeroshot':
        base_config_path = configPathManager.extract_config_path(
            file_name=config_file_name, 
            sub_dir=os.path.join('preprocessing', 'datasets')
        )
        base_config = configPathManager.load_config_yaml_path(base_config_path)

    elif prompt_type == 'finetuned':
        base_config_path = configPathManager.extract_config_path(
            file_name=config_file_name, 
            sub_dir=os.path.join('finetuning', 'experiments')
        )
        part_config = configPathManager.load_config_yaml_path(base_config_path)
        
        # Process each dataset configuration
        for dataset_name in main_config.get('dataset', {}).keys():
            # Update missing fields in main config file using corresponding fields from the additional config file
            new_fields_to_add = [
                ['dataset', dataset_name, 'name'],
                ['dataset', dataset_name, 'project_output_dir'],
                ['dataset', dataset_name, 'ml_metadata_file'],
                ['dataset', dataset_name, 'preprocessing_cfg'],
                ['dataset', dataset_name, 'nifti_image_dir'],
                ['dataset', dataset_name, 'nifti_mask_dir'],
                ['dataset', dataset_name, 'mask_labels']
            ]
            default_fields_to_add = [
                ['dataset', 'name'],
                ['project_output_dir'],
                ['ml_metadata_file'],
                ['preprocessing_cfg'],
                ['nifti_image_dir'],
                ['nifti_mask_dir'],
                ['mask_labels']
            ]
            base_config = load_and_merge_dataset_config(part_config, dataset_name, configPathManager, 'preprocessing/datasets', new_fields_to_add, default_fields_to_add)
            del dataset_config

        # Update config with finetuned weights info
        base_config = configPathManager.add_new_config_values(base_config, part_config, ['finetuned_model'], ['prompt_experiment','finetuned_model']) 
        del part_config
    else:
        raise ValueError("Unsupported prompt_type: {}".format(prompt_type))
    
    # Update config with prompt_experiment info
    prompt_experiment_config_name = base_config.get('prompt_experiment', {}).get('config', None)

    if isinstance(prompt_experiment_config_name, str):
        prompt_experiment_config_path = configPathManager.extract_config_path(file_name=prompt_experiment_config_name, 
                                                                       sub_dir=os.path.join('prompting','experiments'))
    else:
        print("No valid experiment 'config' found in 'prompt_experiment'. Proceeding with default prompt experiment.")
        prompt_experiment_config_path = configPathManager.extract_config_path(file_name='baseline_prompt.yaml', 
                                                                       sub_dir=os.path.join('prompting','experiments'))

    prompt_experiment_config = configPathManager.load_config_yaml_path(prompt_experiment_config_path)
    main_config = configPathManager.merge_missing_config_values(prompt_experiment_config, base_config, fields_to_merge='all')  

    return main_config


def load_experiment(config_file_name, base_dir, kwargs=None):
    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=os.path.join('finetuning','experiments'))
    main_config = configPathManager.load_config_yaml_path(config_path)

    # # Apply terminal argument updates
    if kwargs !=None:
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

        # import pdb; pdb.set_trace()
        # dataset = dataset_name

    # # Apply terminal argument updates
    # if kwargs !=None:
    #     apply_overrides2(main_config, kwargs, dataset)

    # import pdb; pdb.set_trace()

    return main_config


def load_evaluation(config_file_name, base_dir, kwargs=None):

    # import pdb; pdb.set_trace()

    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=os.path.join('finetuning','evaluation'))
    main_config = configPathManager.load_config_yaml_path(config_path)

    # import pdb; pdb.set_trace()

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

# ------------- Dynamic Dataset Cfg Updating ------------- #
def update_cfg_for_dataset(cfg, model_cfg):
    """
    Update cfg based on the structure of 'dataset'. If it's a nested dictionary,
    create a modified copy with values replaced for each sub-dictionary in 'dataset'.
    If it's a direct mapping, directly replace the values in cfg.
    
    Parameters:
    - cfg: The configuration dictionary.
    - model_cfg: The current model configuration being processed.
    
    Returns:
    - A list of modified configurations, one for each dataset.
    """
    updated_cfgs = []

    if isinstance(cfg['dataset'], dict):
        # Check if nested dictionary structure
        if all(isinstance(val, dict) for val in cfg['dataset'].values()):
            for dataset_name, dataset_cfg in cfg['dataset'].items():
                updated_cfg = copy.deepcopy(cfg)

                # Replace with actual values for nested structure -> prompting from finetuning yaml -> useful for multidataset case
                updated_cfg['name'] = dataset_cfg['name'] 
                updated_cfg['mask_labels'] = dataset_cfg['mask_labels']
                updated_cfg['project_output_dir'] = dataset_cfg['project_output_dir']

                updated_cfg['input_cfg']['image_path'] = dataset_cfg['nifti_image_dir']
                updated_cfg['input_cfg']['gt_path'] = dataset_cfg['nifti_mask_dir']
                updated_cfg['input_cfg']['ml_metadata_file'] = dataset_cfg['ml_metadata_file']

                updated_cfg['preprocessing_cfg']['image_size'] = dataset_cfg['preprocessing_cfg']['image_size']
                updated_cfg['preprocessing_cfg']['voxel_num_thre2d'] = dataset_cfg['preprocessing_cfg']['voxel_num_thre2d']
                updated_cfg['preprocessing_cfg']['voxel_num_thre3d'] = dataset_cfg['preprocessing_cfg']['voxel_num_thre3d']
                updated_cfg['preprocessing_cfg']['remove_label_ids'] = dataset_cfg['preprocessing_cfg']['remove_label_ids']
                updated_cfg['preprocessing_cfg']['target_label_id'] = dataset_cfg['preprocessing_cfg']['target_label_id']

                updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.project_output_dir", dataset_cfg['project_output_dir'])
                updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", dataset_name)

                updated_cfgs.append(updated_cfg)
        else:
            # Replace directly if cfg['dataset'] has direct mapping -> single dataset / prompting from dataset.yaml directly
            updated_cfg = copy.deepcopy(cfg)
            updated_cfg['name'] = cfg['dataset']['name']
            updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", cfg['dataset']['name'])
            updated_cfgs.append(updated_cfg)

    else:
        raise ValueError("The structure of 'dataset' in the configuration is not supported.")

    # Replace common placeholders outside the dataset-specific loop.
    for updated_cfg in updated_cfgs:
        updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("${experiment.name}", updated_cfg['experiment']['name'])
        updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("${models.model_weights}", model_cfg['model_weights'])
        del updated_cfg['dataset']

    return updated_cfgs





# def load_prompting_experiment(config_file_name, base_dir, prompt_type='zeroshot'):
#     configPathManager = ConfigPathManager(base_dir=base_dir)

#     # Conditionally set base_config_path based on prompt_type
#     if prompt_type == 'zeroshot':
#         data_config_path = configPathManager.extract_config_path(
#             file_name=config_file_name, 
#             sub_dir=os.path.join('preprocessing', 'datasets')
#         )
#         data_config = configPathManager.load_config_yaml_path(data_config_path)

#         if not isinstance(data_config['dataset'], dict):
#             raise ValueError("The structure of 'dataset' in the configuration is not supported.")
        
#         updated_cfgs = []
#         # Check if nested dictionary structure
#         if all(isinstance(val, dict) for val in data_config['dataset'].values()):
#             #TODO
#             for dataset_name, dataset_cfg in data_config['dataset'].items():
#                 updated_cfg = copy.deepcopy(data_config)

#                 # Replace with actual values for nested structure -> prompting from finetuning yaml -> useful for multidataset case
#                 updated_cfg['name'] = dataset_cfg['name'] 
#                 updated_cfg['mask_labels'] = dataset_cfg['mask_labels']
#                 updated_cfg['project_output_dir'] = dataset_cfg['project_output_dir']

#                 updated_cfg['input_cfg']['image_path'] = dataset_cfg['nifti_image_dir']
#                 updated_cfg['input_cfg']['gt_path'] = dataset_cfg['nifti_mask_dir']
#                 updated_cfg['input_cfg']['ml_metadata_file'] = dataset_cfg['ml_metadata_file']

#                 updated_cfg['preprocessing_cfg']['image_size'] = dataset_cfg['preprocessing_cfg']['image_size']
#                 updated_cfg['preprocessing_cfg']['voxel_num_thre2d'] = dataset_cfg['preprocessing_cfg']['voxel_num_thre2d']
#                 updated_cfg['preprocessing_cfg']['voxel_num_thre3d'] = dataset_cfg['preprocessing_cfg']['voxel_num_thre3d']
#                 updated_cfg['preprocessing_cfg']['remove_label_ids'] = dataset_cfg['preprocessing_cfg']['remove_label_ids']
#                 updated_cfg['preprocessing_cfg']['target_label_id'] = dataset_cfg['preprocessing_cfg']['target_label_id']

#                 updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.project_output_dir", dataset_cfg['project_output_dir'])
#                 updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", dataset_name)

#                 updated_cfgs.append(updated_cfg)
#         else:
#             # Replace directly if cfg['dataset'] has direct mapping -> single dataset / prompting from dataset.yaml directly
#             base_config = dict()
#             base_config['name'] = data_config['dataset']['name']

#             new_fields_to_add = [['datasets', data_config['dataset']['name'], 'input_cfg', 'image_path'],
#                                 ['datasets', data_config['dataset']['name'], 'input_cfg', 'gt_path'],
#                                 ['datasets', data_config['dataset']['name'], 'input_cfg', 'ml_metadata_file'],

#                                 ['datasets', data_config['dataset']['name'], 'name'],
#                                 ['datasets', data_config['dataset']['name'], 'mask_labels'],
#                                 ['datasets', data_config['dataset']['name'], 'project_output_dir'],
                                
#                                 ['datasets', data_config['dataset']['name'], 'preprocessing_cfg', 'image_size'],
#                                 ['datasets', data_config['dataset']['name'], 'preprocessing_cfg', 'voxel_num_thre2d'],
#                                 ['datasets', data_config['dataset']['name'], 'preprocessing_cfg', 'voxel_num_thre3d'],
#                                 ['datasets', data_config['dataset']['name'], 'preprocessing_cfg', 'remove_label_ids'],
#                                 ['datasets', data_config['dataset']['name'], 'preprocessing_cfg', 'target_label_id'],
#                                 ['datasets', data_config['dataset']['name'], 'preprocessing_cfg', 'instance_bboxes_flag'],
#                                 ]
#             default_fields_to_add = [['nifti_image_dir'],
#                                     ['nifti_mask_dir'],
#                                     ['ml_metadata_file'],

#                                     ['dataset', 'name'],
#                                     ['mask_labels'],
#                                     ['project_output_dir'],

#                                     ['preprocessing_cfg', 'image_size'],
#                                     ['preprocessing_cfg', 'voxel_num_thre2d'],
#                                     ['preprocessing_cfg', 'voxel_num_thre3d'],
#                                     ['preprocessing_cfg', 'remove_label_ids'],
#                                     ['preprocessing_cfg', 'target_label_id'],
#                                     ['prompt_experiment', 'preprocessing_cfg', 'instance_bboxes_flag'],
#                                     ]
#             import pdb; pdb.set_trace()
#             # Update missing fields in main config file using corresponding fields from the additional config file
#             base_config = configPathManager.add_new_config_values(base_config, data_config, new_fields_to_add, default_fields_to_add)

#             #updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.project_output_dir", dataset_cfg['project_output_dir'])
#             #updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", dataset_name)

#             #updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", cfg['dataset']['name'])
#             #updated_cfgs.append(updated_cfg)

#     elif prompt_type == 'finetuned':
#         base_config_path = configPathManager.extract_config_path(
#             file_name=config_file_name, 
#             sub_dir=os.path.join('finetuning', 'experiments')
#         )
#         part_config = configPathManager.load_config_yaml_path(base_config_path)
        
#         # Process each dataset configuration
#         for dataset_name in main_config.get('dataset', {}).keys():
#             # Update missing fields in main config file using corresponding fields from the additional config file
#             new_fields_to_add = [
#                 ['dataset', dataset_name, 'name'],
#                 ['dataset', dataset_name, 'project_output_dir'],
#                 ['dataset', dataset_name, 'ml_metadata_file'],
#                 ['dataset', dataset_name, 'preprocessing_cfg'],
#                 ['dataset', dataset_name, 'nifti_image_dir'],
#                 ['dataset', dataset_name, 'nifti_mask_dir'],
#                 ['dataset', dataset_name, 'mask_labels']
#             ]
#             default_fields_to_add = [
#                 ['dataset', 'name'],
#                 ['project_output_dir'],
#                 ['ml_metadata_file'],
#                 ['preprocessing_cfg'],
#                 ['nifti_image_dir'],
#                 ['nifti_mask_dir'],
#                 ['mask_labels']
#             ]
#             base_config = load_and_merge_dataset_config(part_config, dataset_name, configPathManager, 'preprocessing/datasets', new_fields_to_add, default_fields_to_add)
#             del dataset_config

#         # Update config with finetuned weights info
#         base_config = configPathManager.add_new_config_values(base_config, part_config, ['finetuned_model'], ['prompt_experiment','finetuned_model']) 
#         del part_config
#     else:
#         raise ValueError("Unsupported prompt_type: {}".format(prompt_type))
    
#     # Update config with prompt_experiment info
#     prompt_experiment_config_name = data_config.get('prompt_experiment', {}).get('config', None)

#     if isinstance(prompt_experiment_config_name, str):
#         prompt_experiment_config_path = configPathManager.extract_config_path(file_name=prompt_experiment_config_name, 
#                                                                        sub_dir=os.path.join('prompting','experiments'))
#     else:
#         print("No valid experiment 'config' found in 'prompt_experiment'. Proceeding with default prompt experiment.")
#         prompt_experiment_config_path = configPathManager.extract_config_path(file_name='baseline_prompt.yaml', 
#                                                                        sub_dir=os.path.join('prompting','experiments'))

#     prompt_experiment_config = configPathManager.load_config_yaml_path(prompt_experiment_config_path)
#     main_config = configPathManager.merge_missing_config_values(prompt_experiment_config, base_config, fields_to_merge='all')  

#     return main_config




# def update_cfg_for_dataset(cfg, model_cfg):
#     """
#     Update cfg based on the structure of 'dataset'. If it's a nested dictionary,
#     create a modified copy with values replaced for each sub-dictionary in 'dataset'.
#     If it's a direct mapping, directly replace the values in cfg.
    
#     Parameters:
#     - cfg: The configuration dictionary.
#     - model_cfg: The current model configuration being processed.
    
#     Returns:
#     - A list of modified configurations, one for each dataset.
#     """
#     updated_cfgs = []

#     if isinstance(cfg['dataset'], dict):
#         # Check if nested dictionary structure
#         if all(isinstance(val, dict) for val in cfg['dataset'].values()):
#             for dataset_name, dataset_cfg in cfg['dataset'].items():
#                 updated_cfg = copy.deepcopy(cfg)

#                 # Replace with actual values for nested structure -> prompting from finetuning yaml -> useful for multidataset case
#                 new_fields_to_add = [['input_cfg', 'image_path'],
#                                     ['input_cfg', 'gt_path'],
#                                     ['input_cfg', 'ml_metadata_file'],
#                                     ]

#                 default_fields_to_add = [['nifti_image_dir'],
#                                         ['nifti_mask_dir'],
#                                         ['project_output_dir'],
#                                         ['ml_metadata_file'],
#                                         ]

#                 fields_to_merge = [['name'],
#                                     ['mask_labels'],
#                                     ['project_output_dir'],
#                                     ['preprocessing_cfg', 'image_size'],
#                                     ['preprocessing_cfg', 'voxel_num_thre2d'],
#                                     ['preprocessing_cfg', 'voxel_num_thre3d'],
#                                     ['preprocessing_cfg', 'remove_label_ids'],
#                                     ['preprocessing_cfg', 'target_label_id'],
#                                     ]

#                 #updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.project_output_dir", dataset_cfg['project_output_dir'])
#                 #updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", dataset_name)

#                 # Update missing fields in main config file using corresponding fields from the additional config file
#                 #main_config = configPathManager.add_new_config_values(main_config, dataset_config, new_fields_to_add, default_fields_to_add)

#                 #updated_cfgs.append(updated_cfg)
#         else:
#             # Replace directly if cfg['dataset'] has direct mapping -> single dataset / prompting from dataset.yaml directly
#             updated_cfg = copy.deepcopy(cfg)
#             updated_cfg['name'] = cfg['dataset']['name']
#             updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("DATASET.name", cfg['dataset']['name'])
#             updated_cfgs.append(updated_cfg)

#     else:
#         raise ValueError("The structure of 'dataset' in the configuration is not supported.")

#     # Replace common placeholders outside the dataset-specific loop.
#     for updated_cfg in updated_cfgs:
#         updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("${experiment.name}", updated_cfg['experiment']['name'])
#         updated_cfg['base_output_dir'] = updated_cfg['base_output_dir'].replace("${models.model_weights}", model_cfg['model_weights'])
#         del updated_cfg['dataset']

#     return updated_cfgs
