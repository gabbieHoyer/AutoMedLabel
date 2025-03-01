# src/utils/metadata_io.py
import json
import os

def load_json(file_path: str):
    """Load JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, output_path: str):
    """Save the metadata to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    return True

def save_parquet(df, output_path: str):
    """Save data to a parquet file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression='snappy')
    pass

def save_metadata(data, save_path:str):
    """ 
    Save metadata to a parquet or json file.
    """
    #TODO check if metadata exists or force flag

    save_dir = os.path.dirname(save_path)
    # Make output directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if save_path.endswith('.json'):
        return save_json(data, save_path)
    elif save_path.endswith('.parquet'):
        return save_parquet(data, save_path)
    return False