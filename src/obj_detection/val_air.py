import argparse
import os
import yaml
from ultralytics import YOLO
from ultralytics import RTDETR

def load_config(config_file_name, base_dir):
    config_path = os.path.join(base_dir, "configs", config_file_name)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    return config

def load_model(config):
    model_type = config.get('model_type', 'YOLO').upper()  # Default to YOLO if not specified

    if model_type == 'YOLO':
        model_class = YOLO
    elif model_type == 'RTDETR':
        model_class = RTDETR
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model_class


def val_model(config):
    ModelClass = load_model(config)
    model = ModelClass(config['best_weights'])

    metrics = model.val(plots=True)  # Perform validation
    
    # Print metrics for insight
    print(f"mAP50-95: {metrics.box.map}, mAP50: {metrics.box.map50}, mAP75: {metrics.box.map75}")
    print(f"mAPs per category: {metrics.box.maps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO Model with Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    base_dir = os.getcwd()  # Assumes script is run from the project root
    config_name = args.config_name + '.yaml'
    config = load_config(config_name, base_dir)

    val_model(config)


















# from ultralytics import YOLO


# best_weights = "runs/detect/train/weights/best.pt"
# model = YOLO(best_weights)  

# metrics = model.val()  # val the model - # no arguments needed, dataset and settings remembered

# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category

# # documentation: https://docs.ultralytics.com/tasks/detect/#val

