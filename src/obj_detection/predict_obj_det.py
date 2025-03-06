import os
import argparse

from src.obj_detection.core import load_model
from src.utils import load_det_config, get_project_root
root = get_project_root()

def predict_obj_det_impl(config):
    # Load the model class based on configuration and initialize the model using best weights.
    ModelClass = load_model(config)
    model = ModelClass(config['best_weights'])
    
    # Run predictions with parameters from the configuration.
    results = model(
        source=config['data'], 
        conf=config.get('conf', 0.75), 
        classes=config.get('classes', None), 
        max_det=config.get('max_det', 20), 
        save=True, 
        save_txt=True,  
        run_dir=config.get('run_dir', None)
    )
    print(f"Prediction results saved: {results}")

def predict_obj_det(config):
    """
    Public entry point.
    If a configuration filename (string) is provided, it is loaded first.
    """
    if isinstance(config, str):
        base_dir = os.getcwd()  # Assumes the script is run from the project root.
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        config = load_det_config(config_file, base_dir)
    predict_obj_det_impl(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLO Model using Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    predict_obj_det(args.config_name)
