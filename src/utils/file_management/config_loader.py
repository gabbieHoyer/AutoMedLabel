import os
import yaml
from typing import Union
import warnings
warnings.filterwarnings('ignore')
# -------------------- Basic Tools --------------------

def load_yaml(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    return config

def flatten_context(context, parent_key='', sep='.'):
    items = []
    for k, v in context.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_context(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_config_values(config):

    # Flatten the configuration context for dot notation support
    flat_config = flatten_context(config)
    
    # Function to substitute placeholders with actual values
    def substitute(current_config):
        if isinstance(current_config, dict):
            for key, value in current_config.items():
                current_config[key] = substitute(value)
        elif isinstance(current_config, list):
            return [substitute(element) for element in current_config]
        elif isinstance(current_config, str):
            # Detect placeholder and substitute with the value from flat_config
            while '${' in current_config:
                start_index = current_config.find('${')
                end_index = current_config.find('}', start_index)
                if start_index == -1 or end_index == -1:
                    break  # No substitution needed
                placeholder = current_config[start_index + 2:end_index]
                if placeholder in flat_config:
                    # Convert the value to a string if it's not already a string
                    substitute_value = str(flat_config[placeholder])
                    # Replace placeholder with actual value
                    current_config = current_config.replace(
                        f'${{{placeholder}}}', substitute_value)
                else:
                    break  # Placeholder not found in flat_config
        return current_config
    
    # Substitute placeholders in the original nested config
    return substitute(config)


# -------------------- Tools to update dictionary values based on keys --------------------

def find_hierarchical_keys_in_dict(input_dict:dict, current_key_path:list=None, all_paths:list=None):
    """
    Recursive function to find all paths to values in a hierarchical dictionary.
    Params:
    - input_dict: The hierarchical dictionary to search.
    - current_key_path: The current path being traversed (used internally). Defaults to None.
    - all_paths: List to store all found paths (used internally). Defaults to None.

    Returns:
    - all_paths: A list where each item is a list of keys to a value.
    """
    if current_key_path is None:
        current_key_path = []
    if all_paths is None:
        all_paths = []

    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            find_hierarchical_keys_in_dict(value, current_key_path + [key], all_paths)
    else:
        all_paths.append(current_key_path)

    return all_paths

def check_valid_hierarchical_keys(data:dict, hierarchical_keys_list:list): # -> list(bool) or bool
    """
    Determine whether hierarchical key exists and has a value (not empty) in hierarchical dictionary.
    Params:
    - data: The hierarchical dictionary to search.
    - hierarchical_key_list:  A list where each item is a hierarchical key. The hierarchical key is a list of keys to traverse to value.
        ex 1.   ['level_1_item_1','level_2_item_1']

        ex 2.   [['level_1_item_1','level_2_item_1', 'level_3_item_1'],
                 ['level_1_item_1','level_2_item_3'],
                 ['level_1_item_2',] ]
    
    Returns:
    - validity: True or False, or list of True or False. Indicates whether key is present and has a value in hierarchical dictionary.
    """

    def hierarchical_keys_exist_in_dict(data:dict, hierarchical_keys:list):
        """
        Recursive function to check if a set of keys exist in a hierarchical dictionary. (Internal usage)
        Params:
        - data (dict): The hierarchical dictionary to search.
        - keys_list (list): List of keys to check for existence.
        
        Returns:
        -bool: True if all keys exist, False otherwise.
        """

        # current_data = data
        # for key in hierarchical_keys:
        #     # Check whether key is valid
        #     if key not in current_data.keys():
        #         return False
        #     # Check whether value exists
        #     elif not current_data[key]:
        #         return False
        #     current_data = current_data[key]
        # return True

        current_data = data
        for key in hierarchical_keys:
            # Check whether key is valid
            if key not in current_data:
                return False
            # Move to next level in the dictionary
            current_data = current_data[key]
        return True  # Only check for the existence of the key, regardless of its value.


    if isinstance(hierarchical_keys_list, list):
        if isinstance(hierarchical_keys_list[0], str):
            return hierarchical_keys_exist_in_dict(data, hierarchical_keys_list)
        elif isinstance(hierarchical_keys_list[0], list):
            validity = []
            for hierarchical_keys in hierarchical_keys_list:
                validity.append(hierarchical_keys_exist_in_dict(data, hierarchical_keys))
            return validity
    
    print("Invalid hierarchical_keys_list")
    return False


    # def hierarchical_keys_exist_in_dict(data: dict, hierarchical_keys: list):
    #     """
    #     Recursive function to check if a set of keys exist in a hierarchical dictionary. (Internal usage)
    #     Params:
    #     - data (dict): The hierarchical dictionary to search.
    #     - keys_list (list): List of keys to check for existence.
        
    #     Returns:
    #     - bool: True if all keys exist, False otherwise.
    #     """

    #     current_data = data
    #     for key in hierarchical_keys:
    #         # Check whether key is valid
    #         if key not in current_data:
    #             return False
    #         # Move to next level in the dictionary
    #         current_data = current_data[key]
    #     return True  # Only check for the existence of the key, regardless of its value.



def merge_missing_keys(dict1:dict, dict2:dict, hierarchical_key:list[str]) -> dict:
    """
    Recursive function to ensure dict1 has a value at the given hierarchical key. Function will 
    traverse the hierarchical key list. If there is a value in dict1, no updates. If dict1 is missing 
    a value, updated dict1 value with the corresponding value (same hierarchical key path) in dict2.
    
    Params:
    - dict1: A hierarchical dictionary that needs a value for hierarchical key.
    - dict2: A hierarchical dictionary with hierarchical key: value.
    - hierarchial_key: List of keys (strings) to traverse to value. 

    Returns:
    - dict1: dict1 with value for each hierarchical key.
    """
    key = hierarchical_key[0]

    # Check for valid key in dict2
    if key not in dict2.keys():
        raise ValueError('Invalid hierarchical_key')
    
    #
    if key in dict1.keys():
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict) and len(hierarchical_key)>1:
            dict1[key] = merge_missing_keys(dict1[key], dict2[key], hierarchical_key[1:])
        # Check whether no value exists
        elif not dict1[key]:
            dict1[key] = dict2[key]
    else:
        dict1[key] = dict2[key]
    return dict1

def extract_values_from_keys(input_dict:dict, hierarchical_keys_list:list):
    def extract_value_from_key(input_dict:dict, hierarchical_key:list):
        dict_value = input_dict.copy()
        for key in hierarchical_key:
            dict_value = dict_value[key]
        return dict_value

    value_list = []
    for hierarchical_key in hierarchical_keys_list:
        value = extract_value_from_key(input_dict, hierarchical_key)
        value_list.append(value)
    return value_list

# NOTE: 'Invalid field to merge.' happens for prompting
def add_new_keys(input_dict:dict, hierarchical_key:list[str], default_value:str) -> dict:
    """
    Recursive function to ensure dict1 has a value at the given hierarchical key. Function will 
    traverse the hierarchical key list. If there is a value in dict1, no updates. If dict1 is missing 
    a value, updated dict1 value with the corresponding value (same hierarchical key path) in dict2.
    
    Params:
    - input_dict: A hierarchical dictionary that needs a value for hierarchical key.
    - dict2: A hierarchical dictionary with hierarchical key: value.
    - hierarchial_key1: List of keys (strings) in input_dict to traverse to value. 
    - hierarchial_key2: List of keys (strings) in dict2 to traverse to value. 

    Returns:
    - dict1: dict1 with value for each hierarchical key.
    """
    current_level = input_dict
    # Traverse keys to the field value considering cases where the key does and does not exist in
    for key in hierarchical_key[:-1]:
        if key not in current_level:
            current_level[key] = {}
        current_level = current_level[key]

    # Once we have arrived at the last key to field, update the dictionary if no value exists
    # If the key does not exist, or the value does not exist ...
    if not (hierarchical_key[-1] in input_dict.keys()) or not (input_dict[hierarchical_key[-1]]):
        current_level[hierarchical_key[-1]] = default_value

    return input_dict


# --------------- Main Class to load and manipulate config files ---------------
    
class ConfigPathManager():
    """
    Class to manage configs paths and extract info. 
    """
    def __init__(self, base_dir:str):
        """
        Params:
        - base_dir: directory containing all 'config' files for project.
        """
        self.base_dir = os.path.join(base_dir, 'config')
        self.config_cache = {}  # Cache for loaded configurations

    def extract_config_path(self, file_name:str, sub_dir:str=None):
        """Return the full path to the config file."""

        config_path = os.path.join(self.base_dir, 
                                   sub_dir, 
                                   file_name)
        
        if not os.path.exists(config_path):
            raise ValueError(f'Invalid full path to configuration file: {config_path}')
        return config_path
    
    # def load_config_yaml_path(self, config_path:str):
    #     """ Load yaml and preprocess contents."""
    #     config = load_yaml(config_path)
    #     processed_config = process_config_values(config)
    #     return processed_config
    
    # for improved memory use for large-scale config merge (needed for complex experimentations)
    def load_config_yaml_path(self, config_path:str):
        """Load yaml and preprocess contents with caching."""
        if config_path not in self.config_cache:
            config = load_yaml(config_path)
            processed_config = process_config_values(config)
            self.config_cache[config_path] = processed_config
        return self.config_cache[config_path]

    def merge_missing_config_values(self, current_config:dict, default_config:dict, fields_to_merge:Union[dict,str]) -> dict:
        """
        Ensure the current configuration has values for all the specified fields. If a value is missing in the current
        config, update with the corresponding value from the default configuration.
        Supports nested hierarchical config keys and checks whether value is present even if key exists.

        Params:
        - current_config: current configuration that requires specified fields
        - default_config: default configuration will values for specified fields if needed
        - fields_to_merge: 'all' or a list where each item is a hierarchical key. The hierarchical key is a list of keys to traverse to value.
            ex 1.   ['level_1_item_1','level_2_item_1']

            ex 2.   [['level_1_item_1','level_2_item_1', 'level_3_item_1'],
                     ['level_1_item_1','level_2_item_3'],
                     ['level_1_item_2',] ]
        """
        # Determine which fields to merge
        if fields_to_merge == 'all':
            valid_fields_to_merge = find_hierarchical_keys_in_dict(default_config)
        else:
            valid_fields_to_merge = [hierarchical_key for hierarchical_key in fields_to_merge \
                                        if check_valid_hierarchical_keys(default_config, hierarchical_key) is True]
        # If only one field to merge, match datatype with multiple fields to merge
        if not isinstance(valid_fields_to_merge[0], list):
            valid_fields_to_merge = [valid_fields_to_merge]
        
        # Iterate over the fields to merge, check if they're not in the experiment config
        for hierarchical_key in valid_fields_to_merge:
            if check_valid_hierarchical_keys(current_config, hierarchical_key) == False:
                # Copy the field from dataset config to experiment config
                current_config = merge_missing_keys(current_config, default_config, hierarchical_key)

        # Update config with processed values
        processed_config = process_config_values(current_config)
        return processed_config

    
    def add_new_config_values(self, current_config:dict, default_config:dict, new_fields_to_add:Union[dict,str], default_fields_to_add:Union[dict,str]) -> dict:
        """
        Ensure the current configuration has values for all the specified fields. If a value is missing in the current
        config, update with the corresponding value from the default configuration.
        Supports nested hierarchical config keys and checks whether value is present even if key exists.

        Params:
        - current_config: current configuration that requires specified fields
        - default_config: default configuration will values for specified fields if needed
        - new_fields_to_add: a list where each item is a hierarchical key. The hierarchical key is a list of keys to traverse to value.
            ex 1.   ['level_1_item_1','level_2_item_1']
            ex 2.   [['level_1_item_1','level_2_item_3', 'level_3_item_1'],
                     ['level_1_item_2'] ]
        - default_fields_to_add: a list where each item is a hierarchical key. The hierarchical key is a list of keys to traverse to value.
            ex 1.   ['level_1_item_1','level_2_item_1']
            ex 2.   [['level_1_item_1','level_2_item_3', 'level_3_item_1'],
                     ['level_1_item_2',] ]
        """
        # If only one field to merge, match datatype with multiple fields to merge

        if not isinstance(new_fields_to_add[0], list):
            new_fields_to_add = [new_fields_to_add]
        if not isinstance(default_fields_to_add[0], list):
            default_fields_to_add = [default_fields_to_add]
        # Extract default_config dictionary with only key-values to add
        valid_new_fields = []
        valid_default_fields = []

        # bug happens when the value trying to pull from previous config is a flag set to False...
        for new_hierarchical_key, default_hierarchical_key in tuple(zip(new_fields_to_add, default_fields_to_add)):
            # import pdb; pdb.set_trace()

            if check_valid_hierarchical_keys(default_config, default_hierarchical_key) is True:
                valid_new_fields.append(new_hierarchical_key)
                valid_default_fields.append(default_hierarchical_key)
            else:
                raise UserWarning(f'Default key does not exist in default dict. Unable to add. {default_hierarchical_key}')
        

        # If only one field to merge, match datatype with multiple fields to merge
        if not isinstance(valid_new_fields[0], list):
            valid_new_fields = [valid_new_fields]
        if not isinstance(valid_default_fields[0], list):
            valid_default_fields = [valid_default_fields]
        # Extract values for new fields
        relevant_default_values = extract_values_from_keys(default_config, valid_default_fields)

        # Iterate over the fields to merge, check if they're not in the experiment config
        for new_hierarchical_key, default_value in tuple(zip(valid_new_fields, relevant_default_values)):
            if check_valid_hierarchical_keys(current_config, new_hierarchical_key) == False:
                # Copy the field from dataset config to experiment config
                current_config = add_new_keys(current_config, new_hierarchical_key, default_value)
        
        # Update config with processed values
        processed_config = process_config_values(current_config)
        return processed_config

