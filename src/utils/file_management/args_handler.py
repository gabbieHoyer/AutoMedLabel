import yaml

# def process_kwargs(kwargs_list):
#     """
#     Process a list of key=value strings and return a dictionary with the keys and values.
#     Args:
#         kwargs_list (list): List of strings in the format "key=value".
#     Returns:
#         dict: Dictionary with keys and values correctly typed (int, bool, etc.).
#     """
#     kwargs = {}
#     if kwargs_list:
#         for arg in kwargs_list:
#             key, value = arg.split('=')
#             # Use yaml.safe_load to automatically convert types
#             kwargs[key] = yaml.safe_load(value)
#     return kwargs

# def update_nested_config(config, key_path, value):
#     """
#     Update a nested dictionary using a path (keys separated by dots).
    
#     Args:
#         config (dict): The dictionary to update.
#         key_path (str): The dot-separated path to the key.
#         value: The value to set at the path.
#     """
#     keys = key_path.split('.')
#     current = config
#     for key in keys[:-1]:
#         if key not in current:
#             current[key] = {}
#         current = current[key]
#     current[keys[-1]] = value


def process_kwargs(kwargs_list):
    """
    Process a list of key=value strings and return a dictionary with the keys and values.
    Args:
        kwargs_list (list): List of strings in the format "key=value".
    Returns:
        dict: Dictionary with keys and values correctly typed (int, bool, etc.).
    """
    kwargs = {}
    if kwargs_list:
        for arg in kwargs_list:
            key, value = arg.split('=')
            # Use yaml.safe_load to automatically convert types
            kwargs[key] = yaml.safe_load(value)
    return kwargs

def update_nested_config(config, key_path, value):
    """
    Update a nested dictionary using a path (keys separated by dots).
    
    Args:
        config (dict): The dictionary to update.
        key_path (str): The dot-separated path to the key.
        value: The value to set at the path.
    """
    keys = key_path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def apply_overrides2(config, overrides, dataset_name):
    """
    Apply overrides from command line to the configuration dictionary.
    
    Args:
        config (dict): The original configuration dictionary.
        overrides (dict): The overrides dictionary with paths as keys.
        dataset_name (str): The dataset name to replace if the key is 'dataset_name'.
    """
    for key_path, value in overrides.items():
        # Split the key path into segments
        keys = key_path.split('.')
        # Replace 'dataset_name' with the actual dataset_name
        keys = [dataset_name if key == 'dataset_name' else key for key in keys]
        # Join the keys back into a key_path
        new_key_path = '.'.join(keys)
        update_nested_config(config, new_key_path, value)

# example use for slurm job or script use:
# python main.py experiment_config.yaml --kwargs datamodule.batch_size=32 "datamodule.augmentation_pipeline.color=0.5"
#  key-path should be provided in quotes if it contains special characters or spaces.

def apply_overrides(config, overrides):
    """
    Apply overrides from command line to the configuration dictionary.
    
    Args:
        config (dict): The original configuration dictionary.
        overrides (dict): The overrides dictionary with paths as keys.
    """
    for key_path, value in overrides.items():
        update_nested_config(config, key_path, value)
