# Auto Labeling Script (`auto_label.py`)

## Getting Started

### Configuring the Auto Labeling Script

The `auto_label.py` script automates image labeling by integrating an object detection model with a SAM-variant model (such as SAM, MedSAM, SAM2, or a fine-tuned version). To efficiently utilize this script, follow the structured steps below:

#### 1. Configure Your YAML File

Before running the script, you must specify your settings in a YAML configuration file. This file contains critical parameters for the script, including:

- **Object Detection Model Settings**
- **Segmentation Model Settings**
- **Data Paths**
- **Inference Parameters**

Create or modify a YAML configuration file and ensure it is located in the following directory:

```plaintext
config/obj_detection/inference/your_config_name.yaml
```

Replace your_config_name with the actual name of your configuration file.

#### 2. Navigate to the Root Directory
After configuring your YAML file, navigate to the root directory of the codebase:

```bash
$ cd AutoMedLabel
```

#### 3. Run the Script
Once in the root directory, execute the script by referring to your specific configuration file:

```bash
$ python src/obj_detection/auto_label.py your_config_name
```

Replace your_config_name with the name of your configuration file, omitting the .yaml extension.

**Note**:
The script is optimized for single GPU use only at this time

#### Example
Here is an example of how to run the script with a sample configuration file named OAI_T1_Thigh:

```bash
$ python src/obj_detection/auto_label.py examples/OAI_T1_Thigh
```

**Note**: This assumes OAI_T1_Thigh.yaml is properly placed within the config/obj_detection/inference/ directory.

```css
This revised version separates configuration from execution, emphasizes setting up the YAML file first, and simplifies the instructions for better clarity.
```

