import os
import argparse

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.obj_detection.core import load_model
from src.utils import load_det_config

def validate_obj_det_impl(config):
    ModelClass = load_model(config)
    model = ModelClass(config['best_weights'])
    
    metrics = model.val(
        data=config['data_yaml'],
        plots=True, 
        conf=config.get('conf', 0.5), 
        max_det=config.get('max_det', 20), 
        save_json=config.get('save_json', False), 
        save_hybrid=config.get('save_hybrid', False), 
        run_dir=config.get('run_dir', None)
    )  
    
    print(f"mAP50-95: {metrics.box.map}, mAP50: {metrics.box.map50}, mAP75: {metrics.box.map75}")
    print(f"mAPs per category: {metrics.box.maps}")

def validate_obj_det(config):
    """
    Public entry point. If a configuration filename (string) is provided,
    it is loaded before invoking the internal implementation.
    """
    if isinstance(config, str):
        base_dir = os.getcwd()  # Assumes the script is run from the project root.
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        config = load_det_config(config_file, base_dir)
    validate_obj_det_impl(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO Model with Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    validate_obj_det(args.config_name)
