# Ultralytics Model Scripts for AutoMedLabel

## Overview

This document details the usage of three core scripts designed for the training, validation, and prediction of Ultralytics models (YOLOv8, DETR, etc.) within the AutoMedLabel project.

### Configuration Files

Each script requires a YAML configuration file located in:

```plaintext
config/obj_detection/experiments
```

These configuration files contain parameters such as model details, training epochs, image size, batch size, data paths, and specific settings for training, validation, and prediction.

Here is an example of what a typical YAML configuration file might look like:

```yaml
model: YOLO8/yolov8n.pt
model_type: YOLO
data_yaml: /path/to/your/dataset_yaml_file.yaml
imgsz: 1024
epochs: 100 for full training
batch: 8
rect: False
workers: 2
run_dir: /path/to/your/work_dir/obj_detection/<experiment_name>/runs

# Augmentation settings to account for asymmetric labels in ground truth
fliplr: 1.0
translate: 0.05
scale: 0.4
copy_paste: 0.1

# Validation settings
save_json: True
save_hybrid: True
conf: 0.7

# Prediction settings
best_weights: /path/to/your/work_dir/obj_detection/<experiment_name>/runs/<run_id>/weights/best.pt

classes: [4, 5, 6, 7, 8, 9]
max_det: 10
data: standardized_data/yolo_not_resized_npy/test/images/subject4175-007.npy

```

### Dataset YAML Configuration
Before executing the main experiment scripts, it is necessary to create and configure the dataset YAML file referenced by the data_yaml field in your experiment YAML file. This file organizes your training, validation, and test datasets for the models.

Location and Example of Dataset YAML File
Create or modify a dataset YAML file and ensure it is located in:

```plaintext
config/obj_detection/datasets
```

Here is an example of what a dataset YAML file might look like:

```yaml
names:
- background
- sc_fat
- fascia
- extensors
- hamstrings
- fem_cortex
- fem_bm
- adductors
- sartorius
- gracilis
- nv

nc: 11

path: /path/to/your/data/standardized_data/yolo_square_resized

train: train/images
val: val/images
test: test/images
```

Ensure this file is correctly referenced in your main experiment configuration file under the `data_yaml` field.

### Scripts Usage
#### 1. Training Script (train_msk.py)
Run the training script from the root directory with the appropriate YAML configuration file:

```bash
$ python src/obj_detection/train_msk.py <yaml file name>
```

#### 2. Validation Script (val_msk.py)
Execute the validation script after training to evaluate the model's performance:

```bash
$ python src/obj_detection/val_msk.py <yaml file name>
```

#### 3. Prediction Script (predict_msk.py)
Use the prediction script to generate predictions using the trained model:

```bash
$ python src/obj_detection/predict_msk.py <yaml file name>
```

**Note**
Make sure to replace `<yaml file name>` with the actual name of your YAML configuration file without the `.yaml` extension. For instance, if your YAML file is named exp1_obj_det.yaml, the command would be:

```bash
$ python src/obj_detection/train_msk.py exp1_obj_det
```

Additional Information
- These scripts follow the documentation and standards set by Ultralytics.
- Ensure that the YAML configuration files are correctly formatted and located in the specified directory.
- Adjust the paths and parameters according to your specific setup and requirements.

```css
This comprehensive guide now includes the creation and management of dataset YAML files along with instructions for running the main scripts, ensuring users have all necessary details to operate within the framework efficiently.
```