# Dynamic Data Loader for SAM/SAM2 Models
## Overview
The data loader in the **SAM/SAM2 model finetuning** and **evaluation pipeline** provides flexibility for handling multiple datasets and their associated metadata. This document outlines how to set up your datasets for finetuning experiments, control the number of subjects from each dataset, handle imbalanced datasets, and apply augmentations.

The data loader supports:

- **Single dataset** or **multi-dataset** experiments.
- **Subject selection control** for limiting the number of subjects used.
- **Dynamic balancing** for imbalanced datasets (datasets with uneven slice/subject counts).
- **Custom augmentations**, including bounding box shifts and advanced transformations.

## Configuring Datasets in Finetuning Experiments
The `dataset` section in your YAML configuration file specifies which datasets will be used for finetuning. Each dataset points to its metadata location, ensuring the right data is pulled during training, validation, and testing.

**Example 1: Single Dataset Finetuning**

```bash
dataset:
  OAI-thigh:
    config: examples/OAI_T1_Thigh_plus.yaml
```

In this example, only the **OAI-thigh** dataset is used for finetuning. The path to its metadata is defined in the `config` field.

**Example 2: Multi-Dataset Finetuning**

```bash
dataset:
  TBrecon:
    config: TBrecon.yaml
  P50-MAPSS:
    config: P50_compart.yaml
  AFCL-MAPSS:
    config: AFACL_compart.yaml
  OAI-imorphics:
    config: OAI_imorphics.yaml
  OAI-thigh:
    config: OAI_T1_Thigh.yaml 
  KICK-hip:
    config: KICK_cube.yaml
```

Here, multiple datasets (e.g., **TBrecon**, **P50-MAPSS**, **AFCL-MAPSS**, etc.) are mixed together for finetuning. Each dataset is referenced through a separate `config` file that points to the metadata required for training.

## Metadata Configurations for Datasets
Each dataset configuration file (located in `config/preprocessing/datasets/`) must provide paths to important metadata files and directories. These include information about slice-level statistics, subject-level metadata, and paths to preprocessed image and ground truth mask files in parquet format.

**Example Metadata Configuration for a Dataset**

```yaml
# Operation A - location for slice-level statistics
stats_metadata_file: ${general_data_dir}/metadata/volume_metadata_for_stats.json

# Operation B - location of subject-level metadata used for training
ml_metadata_file: ${general_data_dir}/metadata/volume_metadata_for_ml.json  

# Operation C - location to parquet files for ML (paths to preprocessed npy images and ground truth masks compatible with SAM/SAM2 models)
slice_info_parquet_dir: ${general_data_dir}/metadata/SAM2/slice_paths
```

### Breakdown of Metadata Operations:
- **Operation A**: `stats_metadata_file` points to a JSON file containing slice-level statistics such as pixel dimensions, slice thickness, etc.
- **Operation B**: `ml_metadata_file` contains subject-level metadata, including the split sets for training, validation, and testing.
- **Operation C**: `slice_info_parquet_dir` points to the directory where parquet files provide paths to the preprocessed images and ground truth masks used in finetuning or evaluation.

### Configuring the datamodule
The `datamodule` section in the YAML file controls how datasets are processed during finetuning. This includes defining how many subjects to use from each dataset, whether to dynamically balance the datasets, and whether augmentations (like bounding box shifts) should be applied.

#### Key Parameters in `datamodule`
- `max_subject_set`: Controls how many subjects from each dataset will be used. You can set it to a fixed number (e.g., `10`) or use `full` to include all subjects.
- `balanced`: If `True`, the system balances datasets by resampling from those with fewer subjects or slices.
- `bbox_shift`: Applies random bounding box shifts as an augmentation technique.
- `batch_size`: Specifies the batch size for training.
- `num_workers`: Number of workers for data loading.
- `augmentation_pipeline`: Optional field to specify a configuration for custom augmentations.

**Example datamodule Configuration**

```yaml
datamodule:
  max_subject_set: full
  balanced: False
  bbox_shift: 5
  batch_size: 4
  num_workers: 2
  # augmentation_pipeline:
  #   config: simple.yaml
```

In this example:

- `max_subject_set: full`: All subjects from each dataset will be used.
- `balanced: False`: No downsampling will be applied to datasets with more slices; they remain unbalanced.
- `bbox_shift: 5`: Bounding box shifts of 5 pixels are applied as part of augmentation.
- `batch_size: 4`: The training batch size is set to 4.
- `num_workers: 2`: Two workers are used for loading data.


## Dynamic Dataset Mixing and Balancing
### Mixing Multiple Datasets
When multiple datasets are specified in the `dataset` section, the data loader mixes subjects and slices from each dataset into the training, validation, and test sets. The number of subjects from each dataset can be controlled using the `max_subject_set` parameter in the `datamodule` configuration.

- `max_subject_set: full`: Uses all subjects from each dataset, regardless of subject imbalance.
- `max_subject_set: 10`: Limits each dataset to a maximum of 10 subjects.

### Balancing Imbalanced Datasets
If some datasets have significantly more subjects or slices than others, you can enable **dynamic balancing** by setting `balanced: True`. This ensures all datasets contribute equally to the training process by resampling from larger datasets to match the number of slices in smaller datasets.

```yaml
datamodule:
  balanced: True
```

When balancing is enabled, the system calculates a downsampling factor for datasets with more slices, reducing their influence to ensure that smaller datasets are equally represented.

## Saving Dataset Summaries
During the data loading process, the loader generates and saves a summary of the dataset. This summary includes details about the number of subjects, slices, and segmentation labels. Summaries are saved as CSV files for reporting or analysis purposes.

**Example of Saving a Dataset Summary**

```python
save_dataset_summary(summaries, summary_file_path, max_subjects=cfg.get('datamodule', {}).get('max_subject_set', 'full'))
```

Summaries are saved in CSV format and provide detailed insights into the dataset used for training or evaluation.

## Augmentation Support
The data loader allows you to apply custom augmentations during training. These augmentations can range from simple transformations (e.g., flipping, rotating) to advanced methods like brightness/contrast adjustments, grid distortions, and more.

If you need to apply advanced augmentations, you can use the augmentation_pipeline field:

```yaml
augmentation_pipeline:
  config: advanced_augmentation.yaml
```

### Example Augmentation Configuration
The augmentation pipeline configuration files should be placed in the `config/preprocessing/augmentations/` directory. You can define augmentations for your training data, and the system will apply them during data loading.

Below is an example of an augmentation configuration.

**Example 1: Basic Augmentation Pipeline**

```yaml
augmentation_pipeline:
  train:
    - transform: HorizontalFlip
      args:
        p: 0.5
    - transform: VerticalFlip
      args:
        p: 0.5
    - transform: RandomBrightnessContrast
      args:
        brightness_limit: 0.2
        contrast_limit: 0.2
        p: 0.5
```

This example applies:

- **Horizontal Flip**: A 50% chance of flipping images horizontally.
- **Vertical Flip**: A 50% chance of flipping images vertically.
- **Random Brightness/Contrast Adjustment**: Adjusts brightness and contrast within the range of 0.2 with a 50% chance.


## Dataset Classes for Finetuning and Evaluation
### Overview
The dataset classes used in this pipeline enable flexible handling of segmentation masks and bounding box prompts. These dataset classes support both single-class and multi-class segmentation and are compatible with SAM and SAM2 models. For finetuning, a random segmentation mask label is selected for each image, while for evaluation, all segmentation mask labels are used to generate bounding box prompts for prediction.

### Finetuning Dataset Class: `mskSAM2Dataset`
The `mskSAM2Dataset` class is responsible for managing the dataset during the finetuning process. It randomly selects one segmentation mask label from the available ground truth masks (in case of multi-class segmentation), computes the bounding box for the selected label, and applies optional augmentations. This allows for dynamic bounding box prompt generation, which is compatible with SAM and SAM2 models.

#### Key Features:
- **Single Label Selection**: For each image, one segmentation label is randomly chosen from the available ground truth masks.
- **Bounding Box Prompt**: Computes bounding box coordinates for the selected label, with optional bounding box shift augmentations.
- **Augmentations**: Supports custom augmentation pipelines applied to the images and masks.
- **Batches**: The output is batched and includes the image, ground truth mask, bounding box, and other relevant information for the finetuning engine pipeline.


**Example Code:**

```python
class mskSAM2Dataset(Dataset):
    def __init__(self, root_paths, gt2_paths, img_paths, bbox_shift=0, instance_bbox=False, remove_label_ids=[], dataset_name=None, augmentation_config=None):
        self.root_paths = root_paths
        self.gt2_path_files = gt2_paths
        self.bbox_shift = bbox_shift
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids
        self.dataset_name = dataset_name
        self.augmentation_pipeline = build_augmentation_pipeline(augmentation_config)

    def __getitem__(self, index):
        # Loads the image and ground truth mask
        img_1024 = np.load(os.path.join(self.root_paths[index], "imgs", img_name))
        gt = np.load(self.gt2_path_files[index])

        # Randomly selects one label from the ground truth mask
        label_ids = np.unique(gt)[1:]
        chosen_label = random.choice(label_ids)

        # Generates the bounding box for the selected label, with optional perturbation
        y_indices, x_indices = np.where(gt == chosen_label)
        bbox = np.array([np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]) * 4

        return {
            'image': torch.tensor(img_1024).float(),
            'gt2D': torch.tensor(gt == chosen_label).long(),
            'boxes': torch.tensor(bbox).float(),
            'label_id': torch.tensor(chosen_label).long(),
            'img_name': img_name,
            'dataset_name': self.dataset_name
        }
```

**Customization Options:**
- **Bounding Box Shift**: You can control how much the bounding box is shifted during augmentation using the bbox_shift parameter.
- **Augmentations**: Supports an augmentation pipeline passed through the augmentation_config, allowing you to apply custom transformations.

### Evaluation Dataset Class: `MultiClassSAM2Dataset`
The MultiClassSAM2Dataset class is designed for evaluation, particularly when evaluating multiple segmentation mask labels for each image. It computes bounding boxes for all available ground truth segmentation masks and their instances. Additionally, it supports custom biomarker evaluation by including T1rho and T2 maps.

#### Key Features:
Multi-Label Support: Unlike the finetuning dataset, this class generates bounding boxes for all segmentation labels present in the ground truth masks.
Instance Segmentation: Supports instance-based segmentation by computing separate bounding boxes for multiple instances of the same class.
Custom Biomarker Support: Can optionally include T1rho and T2 biomarker maps, which are loaded and included in the evaluation batch.

**Example Code:**

```python
class MultiClassSAM2Dataset(Dataset):
    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, mask_labels=None, instance_bbox=False, remove_label_ids=[], use_biomarkers=False, T1rho_map_paths=None, T2_map_paths=None):
        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids
        self.use_biomarkers = use_biomarkers
        self.T1rho_map_paths = T1rho_map_paths
        self.T2_map_paths = T2_map_paths

    def __getitem__(self, index):
        # Load image and ground truth mask
        img_1024 = np.load(join(self.root_paths[index], "imgs", img_name))
        gt = np.load(self.gt_path_files[index])

        # Generate bounding boxes for all labels in the ground truth mask
        label_ids = np.unique(gt)[1:]
        bbox_list = []
        for label_id in label_ids:
            y_indices, x_indices = np.where(gt == label_id)
            bbox = np.array([np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]) * 4
            bbox_list.append(bbox)

        return {
            'image': torch.tensor(img_1024).float(),
            'gt2D': torch.tensor(gt).long(),
            'boxes': torch.tensor(np.stack(bbox_list)).float(),
            'label_ids': torch.tensor(label_ids).long(),
            'img_name': img_name
        }
```

#### Customization Options:
- **Instance Bounding Boxes**: The class can generate separate bounding boxes for multiple instances of the same class, which is useful in datasets with instance-level segmentation.
- **Biomarker Integration**: You can include custom biomarkers like T1rho and T2 maps in the evaluation batch for downstream analysis, in the case that the biomarker requires an additional file such as T1rho/T2 image maps to compute a tissue value.

#### Advanced Use:
In evaluation, all segmentation labels are evaluated with corresponding bounding boxes, providing a more complete analysis of the model's performance across all regions of interest. This is particularly useful in the evaluation of multi-class models and in downstream biomarker analyses.


## Conclusion
The dynamic data loader in the SAM/SAM2 models pipeline provides a highly flexible and powerful framework for managing complex dataset configurations, subject selection, dynamic balancing, and custom augmentations. Whether your experiments involve single or multiple datasets, the data loader efficiently handles the necessary operations for both finetuning and evaluation, ensuring that bounding box prompts and segmentation labels are processed seamlessly. By leveraging metadata, the loader ensures reproducibility, consistency, and scalability across diverse experiments, making it an essential component for advancing segmentation and biomarker analysis workflows.