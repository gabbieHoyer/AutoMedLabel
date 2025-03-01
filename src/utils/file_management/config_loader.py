# src/utils/file_management/config_loader.py
import os
import yaml
from typing import Any, Dict, List, Union
import warnings

warnings.filterwarnings('ignore')

def load_yaml(config_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    return config

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dictionary using dot notation."""
    items: List[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def substitute_placeholders(config: Any, flat_config: Dict[str, Any]) -> Any:
    """
    Recursively substitute placeholders in the configuration string values.
    Placeholders are in the format ${placeholder}.
    """
    if isinstance(config, dict):
        return {k: substitute_placeholders(v, flat_config) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_placeholders(item, flat_config) for item in config]
    elif isinstance(config, str):
        # While there is at least one placeholder, replace it
        while "${" in config:
            start = config.find("${")
            end = config.find("}", start)
            if start == -1 or end == -1:
                break
            placeholder = config[start+2:end]
            if placeholder in flat_config:
                config = config.replace(f"${{{placeholder}}}", str(flat_config[placeholder]))
            else:
                break  # If not found, stop to avoid infinite loop
        return config
    else:
        return config

def process_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the configuration dictionary and substitute any placeholder values.
    """
    flat = flatten_dict(config)
    return substitute_placeholders(config, flat)

# --- Dictionary Merging Helpers ---
def find_all_key_paths(d: Dict[str, Any], parent: List[str] = None) -> List[List[str]]:
    """Recursively find all key paths in a nested dictionary."""
    if parent is None:
        parent = []
    paths = []
    for key, value in d.items():
        current_path = parent + [key]
        if isinstance(value, dict):
            paths.extend(find_all_key_paths(value, current_path))
        else:
            paths.append(current_path)
    return paths

def keys_exist(d: Dict[str, Any], keys: List[str]) -> bool:
    """Check whether a list of keys exist in a dictionary."""
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True

def merge_dict_value(target: Dict[str, Any], source: Dict[str, Any], key_path: List[str]) -> Dict[str, Any]:
    """
    Ensure that target has the value at key_path; if not, update it from source.
    """
    if not key_path:
        return target
    key = key_path[0]
    if key not in target or not target[key]:
        target[key] = source.get(key)
    elif isinstance(target[key], dict) and isinstance(source.get(key), dict) and len(key_path) > 1:
        merge_dict_value(target[key], source[key], key_path[1:])
    return target

def merge_missing_config_values(current: Dict[str, Any], default: Dict[str, Any],
                                fields: Union[str, List[List[str]]]) -> Dict[str, Any]:
    """
    For each hierarchical key path (as a list of strings) in fields (or 'all' for every key in default),
    ensure that current has that value; if missing, copy it from default.
    """
    if fields == 'all':
        fields = find_all_key_paths(default)
    for key_path in fields:
        if not keys_exist(current, key_path):
            merge_dict_value(current, default, key_path)
    return process_config_values(current)

def add_new_config_values(
    current: Dict[str, Any],
    default: Dict[str, Any],
    new_fields: List[List[str]],
    default_fields: List[List[str]]
) -> Dict[str, Any]:
    """
    For each pair (new_key_path, default_key_path), if `current` is missing `new_key_path`,
    then copy the value from `default` at `default_key_path` into `current[new_key_path]`.
    """
    for new_key_path, default_key_path in zip(new_fields, default_fields):
        if not keys_exist(current, new_key_path):
            # 1) Extract the actual value from `default` using `default_key_path`
            val = get_nested_value(default, default_key_path)
            # 2) If that path existed in `default`, set it into `current` at `new_key_path`
            if val is not None:
                set_nested_value(current, new_key_path, val)

    return process_config_values(current)


def get_nested_value(d: Dict[str, Any], keys: List[str]) -> Any:
    """Traverse `d` by `keys` and return the final value, or None if any key not found."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def set_nested_value(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Traverse `d` by `keys` and set the final key to `value`, creating dicts as needed."""
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


# --- ConfigPathManager Class ---
class ConfigPathManager:
    """
    Manages configuration file paths and provides caching, loading, and merging utilities.
    """
    def __init__(self, base_dir: str):
        self.base_dir = os.path.join(base_dir, 'config')
        self.config_cache: Dict[str, Dict[str, Any]] = {}

    def get_config_path(self, file_name: str, sub_dir: str = None) -> str:
        """Construct and validate the full path to a configuration file."""
        path = os.path.join(self.base_dir, sub_dir, file_name) if sub_dir else os.path.join(self.base_dir, file_name)
        if not os.path.exists(path):
            raise ValueError(f"Configuration file does not exist: {path}")
        return path

    def load_config_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load a YAML configuration with caching and process placeholder substitutions."""
        if config_path not in self.config_cache:
            config = load_yaml(config_path)
            processed = process_config_values(config)
            self.config_cache[config_path] = processed
        return self.config_cache[config_path]

    def merge_missing_values(self, current: Dict[str, Any], default: Dict[str, Any],
                             fields_to_merge: Union[str, List[List[str]]]) -> Dict[str, Any]:
        """
        Merge missing values from the default configuration into the current configuration.
        """
        return merge_missing_config_values(current, default, fields_to_merge)

    def add_new_values(self, current: Dict[str, Any], default: Dict[str, Any],
                       new_fields: List[List[str]], default_fields: List[List[str]]) -> Dict[str, Any]:
        """
        Add new values from the default configuration into the current configuration where missing.
        """
        return add_new_config_values(current, default, new_fields, default_fields)








