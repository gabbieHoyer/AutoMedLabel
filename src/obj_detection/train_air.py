import os
import yaml
import argparse
from string import Template
from ultralytics import YOLO
from ultralytics import RTDETR


def load_config(config_file_name, base_dir):
    """Loads and processes a YAML configuration file."""
    config_path = os.path.join(base_dir, "configs", config_file_name)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}

    # Process the configuration to replace variables
    for key, value in config.items():
        if isinstance(value, str) and '$' in value:
            # Create a Template and substitute variables
            template = Template(value)
            config[key] = template.safe_substitute(os.environ, **config)

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


def main(config):
    # Load a model based on configuration
    ModelClass = load_model(config)
    model = ModelClass(config["model"])

    # Train the model using parameters from the config
    model.train(data=config['data_yaml'], epochs=config['epochs'], imgsz=config['imgsz'], rect=config['rect'], batch=config['batch'], plots=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Model with Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    # Assuming the script is run from the root directory
    base_dir = os.getcwd()
    config_name = args.config_name + '.yaml'
    config = load_config(config_name, base_dir)

    main(config)


# how to use: python train.py exp1_obj_det


# # **************************************************** #
# this worked
# # Load a model
# # model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# # Train the model
# DATA_YAML = "../../datasets/air_data.yaml"
# model.train(data=DATA_YAML, epochs=300, imgsz=[480, 640], rect=True, batch=16)  # train the model

# # **************************************************** #




# path = model.export(format="onnx")  # export the model to ONNX format

# results for train can be interpreted as seen here:
# https://medium.com/the-modern-scientist/yolov8-training-on-custom-data-3460f922ce86


# object detection
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# segmentation
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# classification
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Oriented object detection (OBB)
# model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)

