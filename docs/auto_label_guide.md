# Auto Labeling Script (`auto_label.py`)

## Getting Started

### Running the Auto Labeling Script

The `auto_label.py` script facilitates automatic image labeling by pairing an object detection model with a SAM-variant model (e.g., SAM, MedSAM, SAM2, or a fine-tuned version). Follow the steps below to execute the script:

#### 1. Navigate to the Root Directory

Ensure you are in the root directory of the codebase:

```bash
$ cd AutoMedLabel
```

#### 2. Run the Script
If a GPU is already in use and SLURM is not required or available, you can run the script using a single GPU:

```bash
$ python src/obj_detection/auto_label.py OAI_T1_Thigh
```

Note: OAI_T1_Thigh refers to a YAML configuration file. This file must be located at config/obj_detection/inference/OAI_T1_Thigh.yaml within the root directory.

#### 3. Customize Configuration
Should you need to use a different configuration file, replace OAI_T1_Thigh with the name of your YAML file (excluding the .yaml extension):

```bash
$ python src/obj_detection/auto_label.py your_config_name
```

Make sure your configuration file is positioned within the config/obj_detection/inference/ directory.

Understanding the Configuration File
The YAML configuration file houses parameters critical for:

- Object Detection Model Settings
- Segmentation Model Settings
- Data Paths
- Inference Parameters
- Modify these parameters as necessary before running the script.