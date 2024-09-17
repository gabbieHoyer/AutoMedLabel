# Metadata for Statistics, Dataset Splits, and Preprocessed Image Paths
This markdown file explains the metadata files used in the pipeline to define dataset splits, prepare statistics for downstream analysis, and provide paths to preprocessed image and segmentation mask `.npy` slice files. These metadata files are key components of the data handling system during training and evaluation.

## Overview of Metadata Files
The pipeline utilizes three primary metadata files, each serving a specific role in the experimental process:

- **Operation A - Statistics Metadata (Slice-Level)**: Provides information for statistical analysis of the dataset at the slice level.
- **Operation B - Subject-Level Metadata (ML)**: Contains subject-level metadata, which defines training/validation/test splits and is used to retrieve subject-specific information during model training and evaluation.
- **Operation C - Preprocessed Slice Paths (Parquet)**: A collection of parquet files that contain paths to the preprocessed .npy slice files (image and mask data) used by the dataset class for training and evaluation.

### Configuration Paths in YAML
Here’s an example of how the paths to these metadata files are defined in the dataset YAML configuration:

```yaml
# Operation A - location for Statistics slice-level information
stats_metadata_file: ${general_data_dir}/metadata/volume_metadata_for_stats.json

# Operation B - location of subject-level metadata used for training
ml_metadata_file: ${general_data_dir}/metadata/volume_metadata_for_ml.json

# Operation C - location to parquet files for ML
slice_info_parquet_dir: ${general_data_dir}/metadata/slice_paths
```

### Operation A: **Statistics Metadata** (volume_metadata_for_stats.json)
The volume_metadata_for_stats.json file provides detailed slice-level information for each subject, which is used for statistical analysis of the imaging data. This metadata includes DICOM attributes and technical specifications about each individual slice, such as pixel spacing, slice thickness, and the spatial position of the slice within the subject’s anatomy.

This metadata is essential for various downstream tasks such as volume estimation, tissue segmentation, and other statistical evaluations that require accurate spatial information.

**Example Structure of Slice-Level Stats Metadata:**

```json
{
  "OAI_902173454": {
    "subject_id": "OAI_902173454",
    "slices": {
      "001": {
        "AccessionNumber": "016610439902",
        "SOPInstanceUID": "1.2.82.9798798.342314.324.2343554.23434",
        "StudyInstanceUID": "1.3.114234.1234135.15315.1351235",
        "SeriesInstanceUID": "1.3.12.13251.12351325.123033",
        "series_desc": "AX_T1_THIGH",
        "study_desc": "OAI^MR^ENROLLMENT^THIGH",
        "TE": 10.0,
        "TR": 600.0,
        "flip_angle": 90.0,
        "ETL": 1,
        "field_strength": 2.89362,
        "scanner_name": "",
        "scanner_model": "Trio",
        "slice_thickness": 5.0,
        "slice_spacing": 5.0,
        "pixel_spacing": [0.9765625, 0.9765625],
        "rows": 256,
        "columns": 512,
        "instanceNumber": 1,
        "slice_location": 75.0,
        "image_position_patient": [-233.53511, -125.0, 75.0]
      },
      "007": {
        "AccessionNumber": "016610439902",
        "SOPInstanceUID": "1.2.82.9798798.342314.324.2343554.23434",
        "StudyInstanceUID": "1.3.114234.1234135.15315.1351235",
        "SeriesInstanceUID": "1.3.12.13251.12351325.123033",
        "series_desc": "AX_T1_THIGH",
        "study_desc": "OAI^MR^ENROLLMENT^THIGH",
        "TE": 10.0,
        "TR": 600.0,
        "flip_angle": 90.0,
        "ETL": 1,
        "field_strength": 2.89362,
        "scanner_name": "",
        "scanner_model": "Trio",
        "slice_thickness": 5.0,
        "slice_spacing": 5.0,
        "pixel_spacing": [0.9765625, 0.9765625],
        "rows": 256,
        "columns": 512,
        "instanceNumber": 7,
        "slice_location": 45.0,
        "image_position_patient": [-233.53511, -125.0, 45.0]
      },
      ...
    }
  }
}
```

**Key Fields:**

- `AccessionNumber`, `SOPInstanceUID`, `StudyInstanceUID`, `SeriesInstanceUID`: Unique identifiers for the imaging session and slices, which are crucial for aligning with the DICOM standard.
- `series_desc` / `study_desc`: Descriptions of the imaging series and study.
- `TE` (Echo Time) / `TR` (Repetition Time): Key imaging parameters that impact the contrast and quality of MRI images.
- `flip_angle`: The angle of the radiofrequency pulse in the MRI sequence.
- `field_strength`: The magnetic field strength of the MRI scanner, typically measured in Tesla (e.g., 3.0T).
- `slice_thickness` / `slice_spacing`: Important spatial properties of the MRI slice.
- `pixel_spacing`: The in-plane resolution of the MRI image.
- `rows` / `columns`: Dimensions of the MRI slice.
- `instanceNumber`: The position of the slice in the full volume.
- `slice_location` / `image_position_patient`: Information about the spatial location of the slice relative to the patient.

Purpose of Slice-Level Metadata
This slice-level metadata allows the pipeline to:
- Evaluate individual slices during model training and testing.
- Perform detailed slice-by-slice analysis for downstream tasks like tissue segmentation and thickness measurement.
- Calculate volumetric properties for custom biomarker metrics based on DICOM attributes like pixel spacing and slice thickness.

By having these details for each slice, the pipeline can ensure accurate statistical analysis and produce reliable, clinically relevant results for each subject.


### Operation B: **Subject-Level Metadata for Training** (volume_metadata_for_ml.json)
This file contains subject-level metadata, including paths to the image and mask files and other relevant subject information. It is primarily used to define the training, validation, and test splits, but it also serves as a resource for downstream biomarker analysis, retrieving essential subject-specific details such as sex, age, weight, anatomical region, and MRI acquisition parameters. These metadata fields are crucial for both model training and the extraction of statistical information during evaluation and biomarker computation.

**Example Structure:**

```json
{
  "TBrecon-01-02-00047": {
    "subject_id": "TBrecon-01-02-00047",
    "image_nifti": "/data/TBrecon3/Users/ghoyer/SAM_data/new_design_tests_TBrecon/tbrecon_nifti_img/TBrecon-01-02-00047.nii.gz",
    "mask_nifti": "/data/TBrecon3/Users/ghoyer/SAM_data/new_design_tests_TBrecon/tbrecon_nifti_mask/TBrecon-01-02-00047.nii.gz",
    "Sex": "F",
    "Age": 42,
    "Weight": 70.31,
    "Dataset": "TBrecon",
    "Anatomy": "knee",
    "num_slices": 196,
    "mask_labels": {
      "0": "background",
      "1": "femoral cartilage",
      "2": "tibial cartilage",
      "3": "patellar cartilage",
      "4": "femur",
      "5": "tibia",
      "6": "patella"
    },
    "field_strength": "3.0",
    "mri_sequence": "3D_CUBE",
    "Split": "test"
  },
  ...
}
```

**Key Fields:**
- **subject_id**: Unique identifier for the subject.
- **image_nifti**: Path to the subject’s MRI image in .nii.gz format.
- **mask_nifti**: Path to the segmentation mask for the MRI image.
**Sex**, **Age**, **Weight**: Demographic information about the subject.
- **Dataset**: Name of the dataset the subject belongs to.
- **Anatomy**: Anatomical region of interest (e.g., knee).
- **num_slices**: Total number of slices in the MRI volume.
- **mask_labels**: Mapping of mask labels to anatomical structures.
- **field_strength**: MRI field strength in Tesla (e.g., 3.0T).
- **mri_sequence**: Type of MRI sequence used (e.g., 3D_CUBE).
- **Split**: Indicates whether the subject belongs to the train, validation, or test split.

**Usage:**
This metadata serves a dual purpose:

1. Data Loading and Experimentation: During model training and evaluation, it is used to load and filter subjects into appropriate splits (train, val, test). Each subject's demographic and anatomical information is accessed via the datamodule to ensure proper dataset configuration for fine-tuning and evaluation.

2. Biomarker and Statistical Analysis: The same metadata can be leveraged during biomarker analysis. Information like sex, age, weight, field strength, and MRI sequence is critical for analyzing correlations between model performance and subject-specific factors. For example, in downstream analysis, biomarkers such as cartilage thickness or tissue volume can be computed using this metadata to assess the impact of MRI characteristics or anatomical region on these measurements.

This metadata, therefore, plays a key role in both the training pipeline and statistical evaluations, enabling a smooth transition between model experimentation and in-depth analysis of biomarker metrics.


### Operation C: **Preprocessed Slice Paths** (Parquet Files)
The slice_info_parquet_dir directory contains parquet files that store paths to preprocessed .npy slice files for both images and masks. These parquet files ensure fast loading during model training and evaluation.

**Example Structure of a Parquet File for a Single Subject:**

| npy_image_path | npy_mask_path | npy_base_dir                        |
|----------------|---------------|-------------------------------------|
| img_001.npy    | mask_001.npy   | /data/preprocessed/TBrecon-00047    |
| img_002.npy    | mask_002.npy   | /data/preprocessed/TBrecon-00047    |
| ...            | ...           | ...                                 |


**Key Fields:**
- `npy_image_path`: Path to the preprocessed `.npy` file containing the image data.
- `npy_mask_path`: Path to the .npy file containing the segmentation mask.
- `npy_base_dir`: Base directory where the .npy files are located.

These parquet files enable efficient loading of data slices during model training and evaluation, especially when working with large datasets.


**Example Usage in the Pipeline**

In the pipeline, these metadata files are referenced and utilized during data loading, preprocessing, and model evaluation:

```python
# Set up data loaders using the dataset configuration
train_loader, val_loader, test_loader = datamodule(cfg, run_path)
```

During the execution of the datamodule function, the subject-level metadata and parquet paths are loaded:

```python
dataset_cfg = dict()
for dataset_name in cfg.get('dataset').keys():
    dataset_cfg[dataset_name] = {
        'ml_metadata_file': cfg.get('dataset').get(dataset_name).get('ml_metadata_file'),
        'slice_info_parquet_dir': cfg.get('dataset').get(dataset_name).get('slice_info_parquet_dir'),
        'mask_labels': cfg.get('dataset').get(dataset_name).get('mask_labels'),
        'instance_bbox': cfg.get('dataset').get(dataset_name).get('instance_bbox'),
        'remove_label_ids': cfg.get('dataset').get(dataset_name).get('remove_label_ids')
    }
```

This information is then processed to generate data loaders for training, validation, and testing, ensuring the proper subject-specific data and preprocessed slices are used in the experimentation process.


This document serves as a guide for understanding the metadata structure and how it integrates with the model training and evaluation pipeline.