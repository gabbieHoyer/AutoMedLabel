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

def predict(config):

    ModelClass = load_model(config)
    model = ModelClass(config['best_weights'])

    results = model(source=config['data'], conf=config.get('conf', 0.5), save=True, save_txt=True)
    print(f"Prediction results saved: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLO Model using Config File")
    parser.add_argument("config_name", help="Name of the YAML configuration file without extension")
    args = parser.parse_args()

    base_dir = os.getcwd()  # Assumes the script is run from the project root
    config_name = args.config_name + '.yaml'
    config = load_config(config_name, base_dir)

    predict(config)










# # *************************************************************************
# this worked
# best_weights = "runs/detect/train/weights/best.pt"
# model = YOLO(best_weights)  

# # data = "datasets/preprocessed_data/yolo_format/test/images"
# # data = "datasets/intubation_videos/Challenging_intubation3.mp4"
# # data = "datasets/intubation_videos/Anesthesia First Time Users of VividTrac.mp4"
# data = "datasets/intubation_videos/20201012-150355.avi"

# results: list = model(source=data, conf=0.5, save=True, save_txt=True)  # generator of Results objects

# # *************************************************************************


# metrics = model.predict(source=data, conf=0.25)  #  no arguments needed, dataset and settings remembered

# results = model(source=data, stream=True)  # generator of Results objects
# for r in results:
#     boxes = r.boxes  # Boxes object for bbox outputs
#     print(boxes)
    # masks = r.masks  # Masks object for segment masks outputs
    # probs = r.probs  # Class probabilities for classification outputs

# ********************************************************************************
# # Output an image with the detection results drawn and in text
# model.predict("https://ultralytics.com/images/bus.jpg", save=True, save_txt=True)

# # Output the detection result and the confidence of each object in the text 
# model.predict("https://ultralytics.com/images/bus.jpg", save_txt=True, save_conf=True)

# multiple inputs:
# source_list: list = ["./sample1.jpg", "./sample2.jpg"]
# result: list = model.predict(source_list, save=True)