## Configuration File Fields Guide
This guide provides detailed explanations of the fields within the `dataset.yaml` configuration files used in the mskSAM project. These configuration files are crucial for directing the data processing scripts, including data standardization, metadata creation, and preparation for the Segment Anything Model (SAM).

### General Structure
The `.yaml` file contains several key sections, each related to different stages of data processing and analysis. Below is an overview of these sections and their respective fields.

### ------------------ PARAMS FOR ALL ------------------

- **`overwrite_existing_flag`** (optional): True or False. Specifies whether to overwrite existing files.
- **`default_dataset_suffix_dirs`**: Subdirectories for dataset.
- **`general_data_dir`**: Outputs for standardized data (3D volumes) that use the default folder & file structure.
- **`project_data_dir`**: Outputs for scratch data (2D slices preprocessed for fine tuning SAM) that use the default folder & file structure.
- **`project_output_dir`**: Outputs for inference that use the default folder & file structure. 

Handling DICOM Files Without Standard Extensions
In some datasets, DICOM files may not have the standard `.dcm` extension, which can cause issues during the data processing phase. To ensure these files are recognized and processed as DICOM images, you can use the force_dicom field:

overwrite_existing_flag
- **`no_dicom_extension_flag`** (optional): This field is optional and should be set to True when you want to force the processing of all files in the mri_dicom_dir as DICOM files, regardless of their file extension. This is particularly useful for datasets like OAI (Osteoarthritis Initiative) where DICOM files may lack the conventional `.dcm` extension.

Example for any `/path/to/dicom/images`:

```yaml
force_dicom_flag: True
```
By setting force_dicom to True, you instruct the data processing scripts to treat all files within the specified directory as DICOM files, ensuring that they are included in the metadata extraction and any subsequent processing steps. This ensures comprehensive handling of DICOM datasets, especially those that deviate from typical file naming conventions.

### ------------------ INFO FOR DATA STANDARDIZATION ------------------
**Packaged MRI Image and Mask Folder Paths**
This section specifies the directories and file types for MRI images and segmentation masks, including details for handling datasets where images are not in DICOM format.

- **`image_data_paths`**: The directory containing MRI images (after any initial processing).
- **`mask_data_paths`**: The directory containing segmentation masks.

Handling Non-DICOM File Types for Metadata
For datasets where the images and segmentation masks are stored in formats other than DICOM, additional fields help manage and link back to the original DICOM files for metadata extraction. These configurations are crucial for cases where processed or derived image formats are utilized, ensuring the linkage to original DICOM metadata for comprehensive analysis.

- **`image_column`** (optional): Specifies the column name for paths to the processed image files. This is important for datasets where images are stored in non-DICOM formats (e.g., `.h5`, `.npz`, `.int2`).
- **`mask_column`** or **`mask_columns`** (optional): Specifies the column name(s) for paths to the processed segmentation mask files. This field can accommodate either a single column name (as a string) or a list of column names (as an array) for datasets where multiple mask volumes per subject must be combined into a single file.

    - Single Mask Volume Example:

        ```yaml
        mask_column: erector_spinae_h5
        ```

    - Multiple Mask Volumes Example:

        ```yaml
        mask_columns: ['erector_spinae_h5', 'multifidus_h5', 'psoas_h5', 'quadratus_lumborum_h5']
        ```

In the case of multiple mask volumes, the specified columns will be combined to create a single comprehensive mask file per subject, ensuring that all relevant anatomical structures or regions of interest are included in one file for streamlined processing and analysis.

- **`mask_config`**: 
    - **`image_key`** (optional): The key or label used in the dataset for MRI images.
    - **`mask_key`** (optional): The key or label used for segmentation masks.


Performing data transformations so that the data is in a standardized orientation
```yaml
mask_config:
  transforms:
    transpose: [2, 0, 1]
    flip: 2
    rot90: 
    swapaxes: [0, 1] 
    dtype: np.uint8
```

Enforce that data has given properties before saving.
```yaml
mask_config:
  data_properties:
    finite_values: True
    type: int8
```

Implementation in `data_standardization.py`
The `data_standardization.py` script is designed to handle both scenarios seamlessly. It checks for the existence of either a single mask column or multiple mask columns as specified in the configuration file, and processes the segmentation masks accordingly:

```python
# Pseudo example usage for mask data. Supports both 'mask_columns' and 'mask_column'
columns_to_process = cfg.get("mask_columns", [cfg.get("mask_column")])
data_paths = [row[column].strip() for column in columns_to_process]
# If given one 'mask_column'
if len(data_paths)==1:
    data = get_data(data_paths[0], data_key, nifti_output_dir, force_dicom_flag)
# If given multiple 'mask_columns'
elif len(data_paths)>1:
    data = get_combined_data(data_paths, data_key, nifti_output_dir)
save_nifti(data, save_path)
```

This flexibility allows users to tailor the data standardization process to the specific needs of their dataset, whether it involves simple or complex mask data structures.

**Dataset Details**
Mask labels are dependent on the files loaded and needed for visualization. In 
some cases "mask labels" must correspond to the order the "mask_columns" are defined. 
- **`mask_labels`**: A dictionary mapping numerical labels to anatomical structures or categories within the segmentation masks.
    - Example:
        ```yaml
        mask_labels:
            0: background
            1: anatomy
        ```

**--- OUTPUT (OVERRIDE DEFAULT):**
- **`nifti_image_dir`**: Directory for standardized NIfTI MRI images.
- **`nifti_mask_dir`**: Directory for standardized NIfTI segmentation masks.

### ------------------ NIFTI VISUALIZATION ------------------
- **`nifti_fig_cfg`**: 
    - **`fig_type`** (override default): Specify figure(s) to generate. Visualzation code can plot volume with segmentation overlays as (a) one image that includes subplots of slices or (b) a gif that cycles through slices. Options include '2D_overlay', 'gif', ['2D_overlay', 'gif']
    - **`num_figs`** (override default):  Number of files to randomly select and process
    - **`slice_selection`** (override default):  Specify which files are elidgable for randomly selection. May be 'any' (default) or 'with_segmentation'.

    **--- OUTPUT (OVERRIDE DEFAULT):**
    - **`fig_path`**: location to save figures

### ------------------ INFO FOR SAM PREP ------------------
**SAM Compatible NPY Data Standardization and Slice-Level Metadata Creation**
- **`preprocessing_cfg`**: Configuration to preprocess data for SAM.
    - **`image_size`**: The size to which images should be resized. This is specified as an integer representing the length of one side of the square image, in pixels.
    - **`voxel_num_thre2d`**: Threshold for the number of voxels in 2D. Objects in each slice smaller than this threshold will be removed during preprocessing.
    - **`voxel_num_thre3d`**: Threshold for the number of voxels in 3D. Objects in the entire volume smaller than this threshold will be removed during preprocessing.
    - **`remove_label_ids`**: A list of label IDs to be removed from the mask data. Specify as an empty list `[]` if no labels should be removed. Example: `[1, 2, 3]` for removing labels with IDs 1, 2, and 3.
    - **`target_label_id`**: The label ID of the target area to be segmented into instances. Specify as `null` if instance segmentation should not be targeted towards any specific label. This can be used for various anatomical structures or features that need instance segmentation within medical images, such as bones, cartilage, discs, or muscles. Example: `2` for targeting label ID 2.
    - **`crop_non_zero_slices_flag`**: Whether or not to include slices without segmentations. Set to True for .npz slice standardization.

Please ensure to configure these fields according to your specific preprocessing needs. Leaving `remove_label_ids` as an empty list `[]` or setting `target_label_id` to `null` signifies no action to be taken for those respective preprocessing steps.

**--- OUTPUT (OVERRIDE DEFAULT):** 
- **`npy_dir`**: Directory to output SAM-compatible `.npy` files.

### ------------------ INFO FOR METADATA EXTRACTION ------------------
**DICOM Data Directory and Processing**
- **`mri_dicom_dir`**: Specifies the directory containing original MRI DICOM images. This is where the script looks for DICOM files to extract header information for statistics and other metadata.

- **`dicom_column`** (optional): This field is necessary only if the image_column does not point to DICOM images. It specifies the column name in the metadata CSV or data structure that contains paths or identifiers for the original DICOM images. This allows for the association of processed images and masks with their original DICOM metadata. If image_column already points to DICOM images, then dicom_column is not required and can be considered optional.

This configuration is particularly useful for maintaining a connection to DICOM metadata when working with processed or derived datasets, ensuring that statistical and analytical metadata can still reference the original imaging parameters.

**Dataset Details**
- **`metadata_cfg`**: 
    - **`dataset_name`** (optional): Name of the dataset.
    - **`anatomy`** (optional): Targeted anatomy (e.g., knee, hip).
    - **`field_strength`** (optional): MRI field strength used in the dataset.
    - **`mri_sequence`** (optional): MRI sequence type.

    - **`dicom_values`**: A dictionary of DICOM header fields to extract for statistics metadata, where the key is the standardized name for reference in the project, and the value is the actual DICOM tag name.

    - Example:
        ```yaml
        dicom_values:
            series_desc: SeriesDescription
            study_desc: StudyDescription
            Age: PatientAge   
            Sex: PatientSex
            Weight: PatientWeight
        ```

**Operation A - Optional Table/Dataframe for Demographic or Other Information**
- **`extra_metadata_cfg`** (optional): Configuration for an optional CSV file containing demographic or other data related to the dataset.
    - **`csv_file_path`**: Location of a table/dataframe for demographic or other information desired for metadata
    - **`subject_id_col`**: Column name in the CSV file that contains subject identifiers.
    - **`column_mapping`**: A dictionary mapping standardized field names to actual column names in the CSV file.

    - Additional Metadata CSV Example:

        ```yaml
        csv_config: {
            csv_file_path: /path/to/file.csv
            subject_id_col: 'subject_id'
            column_mapping: {
                'standard name': 'actual name',
                'subject id': 'unique_id',
                }
        }
        ```

**Operation B - Optional Information**
- **`additional_metadata_file`** (optional): JSON dictionary to provide supplementary info (demographics, imaging parameters).

**--- OUTPUT (OVERRIDE DEFAULT):**
**Operation A**
- **`metadata_for_stats_file`**: Path to output the subject-level metadata used for training.
**Operation B**
- **`metadata_for_ml_file`**: Location for Statistics slice-level information
**Operation C**
- **`slice_info_parquet_dir`**: Directory for outputting parquet tables containing slice-level metadata or other structured data. Used for dynamic image resolution and data fusion

