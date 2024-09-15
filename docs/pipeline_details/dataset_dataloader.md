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


## Conclusion
The dynamic data loader in the SAM/SAM2 models pipeline is a versatile tool for handling complex dataset configurations, subject selection, balancing, and augmentations. Whether you're working with a single dataset or multiple datasets, the loader ensures efficient and flexible data handling for training and evaluation. By using metadata, the loader guarantees reproducibility and consistency across experiments.