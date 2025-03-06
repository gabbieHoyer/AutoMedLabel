# src/obj_detection/core.py
import os
import yaml
from string import Template

from src.utils import get_project_root
root = get_project_root()

from src.obj_detection.ultralytics import YOLO, RTDETR, NAS

def load_model(config: dict):
    """
    Return the appropriate model class from ultralytics based on the config.
    """
    model_type = config.get('model_type', 'YOLO').upper()  # Default to YOLO
    if model_type == 'YOLO':
        model_class = YOLO
    elif model_type == 'RTDETR':
        model_class = RTDETR
    elif model_type == 'NAS':
        model_class = NAS
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_class

