import os
import argparse

from src.obj_detection.core import load_model
from src.utils import load_det_config, get_project_root
root = get_project_root()

os.environ['WANDB_MODE'] = 'disabled'

def train_obj_det_impl(config):
    # Load the model class based on configuration.
    ModelClass = load_model(config)
    
    # Resume from checkpoint if available.
    if "checkpoint" in config and os.path.exists(config["checkpoint"]):
        print("Resuming training from checkpoint.")
        model = ModelClass(config["checkpoint"])
    else:
        print("Starting training from pretrained model.")
        model = ModelClass(config["model"])

    # Train the model using parameters from the configuration.
    model.train(
        data=config['data_yaml'], 
        epochs=config['epochs'], 
        imgsz=config['imgsz'], 
        rect=config['rect'], 
        batch=config['batch'], 
        workers=config['workers'], 
        plots=True,  
        run_dir=config.get('run_dir', None)
    )

def train_obj_det(config):
    """
    Public entry point.
    If a configuration filename (string) is provided, it is loaded,
    then the training implementation is invoked.
    """
    if isinstance(config, str):
        base_dir = os.getcwd()
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        config = load_det_config(config_file, base_dir)
    train_obj_det_impl(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Model with Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()
    train_obj_det(args.config_name)
