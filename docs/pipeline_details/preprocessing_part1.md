# Data Standardization Pipeline
The data standardization pipeline is designed to convert various medical image and mask formats into the standardized NIfTI format (`.nii.gz`). This ensures compatibility with downstream tasks such as model training, metadata extraction, and visualization. This pipeline can be skipped if the data is already in NIfTI format.

### Purpose
The data standardization step ensures that:

- MRI image volumes and their associated segmentation masks are converted into a standardized format.
- Data from different file formats (DICOM, NPZ, NIfTI, etc.) can be processed into a single, compatible format for medical imaging tasks.
- Optional transformations (such as transpose, rotation, or connected component analysis) can be applied based on the dataset configuration.

### Key Components
#### Script: `data_standardization.py`
This script handles the conversion of MRI images and masks into NIfTI format. It reads the dataset-specific YAML configuration file to determine which data paths to process, how to handle the data, and whether specific transformations are required.

### Steps Performed
1. Loading Dataset Configurations: The script first loads the dataset-specific YAML configuration file. It retrieves paths to the image and mask data files (which can be in various formats such as DICOM, NPZ, NIfTI, etc.) and applies user-defined transformations if necessary.

2. Converting Data:
  - The convert_to_nifti() function iterates through the provided data paths, processes the MRI image and mask data, and converts it into NIfTI format.
  - If the input data is in DICOM format, additional metadata extraction is performed.
  - The function checks if existing files should be overwritten based on the configuration.

3. Applying Transformations:
  - Transpose: Reorders the axes of the data to match the required format.
  - Flip/Rotate: Flips or rotates the data along specified axes.
  - Connected Component Analysis: For mask data, connected component labeling is performed to isolate specific structures.

4. Saving NIfTI Files:
  - Once processed, the data is saved in NIfTI format with the .nii.gz extension in the specified output directory.

### Configuration Parameters (YAML)
The data standardization script relies on the YAML configuration file to determine how the data should be processed. Here's a breakdown of key parameters used in the config file:

#### Parameter	Description

| Parameter                  | Description                                                             |
|----------------------------|-------------------------------------------------------------------------|
| `image_data_paths`          | Path to the directory or CSV file containing MRI image data.             |
| `mask_data_paths`           | Path to the directory or CSV file containing mask data.                  |
| `image_column`              | Column name in the CSV that holds the image paths.                      |
| `mask_column`               | Column name in the CSV that holds the mask paths.                       |
| `transforms`                | Optional transformations to apply to the data (e.g., transpose, flip).  |
| `nifti_image_dir`           | Output directory where the standardized NIfTI images will be saved.     |
| `nifti_mask_dir`            | Output directory where the standardized NIfTI masks will be saved.      |
| `no_dicom_extension_flag`   | Flag indicating whether DICOM files lack a file extension.              |
| `overwrite_existing_flag`   | Flag indicating whether to overwrite existing NIfTI files.              |


### Example Workflow
1. **Convert MRI Images to NIfTI**: If the dataset contains DICOM or NPZ images, the `convert_to_nifti()` function is used to convert these to NIfTI format.

```bash
python3 data_standardization.py OAI_thigh_muscle
```

- The script looks at the paths defined in the dataset configuration file for image_data_paths and mask_data_paths.
- Images and masks are loaded using the correct format handler (e.g., DICOM, NPZ, etc.).
- Converted NIfTI files are saved in the specified output directory.

2. **Apply Data Transformations**: The script supports optional transformations, such as axis transposition or connected component analysis. These transformations are applied to the data before saving.

Example YAML configuration for transformations:

```yaml
image_config:
  transforms:
    transpose: [2, 0, 1]
    flip: 1
mask_config:
  transforms:
    dtype: np.uint8
    top_cc_3D: 1
```

### Optional: Skipping the Step
If the MRI images and masks are already in NIfTI format, this step can be skipped. 


## NIfTI Visualization for Quality Check
The **NIfTI Visualization** step is an optional quality check that helps visualize MRI images and their corresponding segmentation masks. This step generates either a 2D overlay plot or a GIF of the image slices with the segmentation masks overlaid, allowing the user to verify that the data has been processed correctly.

### Purpose
- Visual Quality Check: The visualization step helps to visually inspect MRI images and their corresponding segmentation masks to ensure that the image and mask data are aligned correctly.
- Figure Generation: Generates either static 2D overlay images or animated GIFs that cycle through the image slices, allowing for a comprehensive review of the data.
- Optional Step: This is an optional step in the preprocessing pipeline and can be skipped if visualization is not required.

### Key Components
#### Script: `nifti_viz.py`
This script handles the visualization of MRI images and segmentation masks in NIfTI format. It loads the processed NIfTI files, generates the specified visualizations (e.g., overlay images or GIFs), and saves them to the specified output directory.

### Steps Performed
1. **Loading Dataset Configurations**: The script first loads the dataset-specific YAML configuration file. It retrieves paths to the NIfTI images and masks, as well as settings for the type of figures to generate (e.g., 2D overlay or GIF).
2. **Visualizing NIfTI Files**: The visualize_nifti() function creates the overlay images or GIFs for a specified number of image and mask pairs. It performs the following tasks:
  - **Overlay Plot Generation**: Creates 2D overlay plots where the segmentation masks are overlaid on the MRI images.
  - **GIF Creation**: Generates an animated GIF that cycles through the slices of the MRI image with the segmentation mask overlay.
3. **Saving the Visualizations**: The generated figures are saved in the specified output directory. The user can choose to overwrite existing visualizations if necessary.

### Configuration Parameters (YAML)
The visualization script relies on the YAML configuration file to determine how the data should be processed. Here's a breakdown of key parameters used in the config file:

#### Parameter	Description

| Parameter                | Description                                                                |
|--------------------------|----------------------------------------------------------------------------|
| `nifti_image_dir`         | Path to the directory containing the NIfTI images.                         |
| `nifti_mask_dir`          | Path to the directory containing the NIfTI masks.                          |
| `nifti_fig_cfg`           | Dictionary containing the figure generation configuration.                 |
| `fig_type`                | Type of figure to generate. Options: 2D_overlay, gif, or both.             |
| `num_figs`                | Number of figures to generate.                                             |
| `slice_selection`         | Criteria for selecting slices (any or with_segmentation).                  |
| `fig_path`                | Path to save the generated figures.                                        |
| `mask_labels`             | Dictionary mapping mask labels to anatomical structures.                   |
| `overwrite_existing_flag` | Flag indicating whether to overwrite existing figures.                     |
| `clim`                    | Color limits for the segmentation overlay (optional).                     |

### Example Workflow
1. **Running NIfTI Visualization**: To visualize the NIfTI images and segmentation masks, run the following command:

```bash
python3 evaluation/visualization/nifti_viz.py OAI_thigh_muscle

```

- This script will generate 2D overlay images or GIFs based on the configuration defined in the YAML file.
- The number of figures generated is controlled by the `num_figs` parameter, and the slices can be selected based on the presence of segmentation using the `slice_selection` parameter.
2. **Figure Generation**: The user can specify the type of figures they want to generate:
- **2D Overlay**: Generates static 2D images with the segmentation masks overlaid on the MRI slices.
- **GIF**: Creates an animated GIF cycling through the slices of the MRI image.

#### Example YAML configuration for generating 2D overlay and GIFs:

```yaml
nifti_fig_cfg:
  fig_type: ['2D_overlay', 'gif']
  num_figs: 5
  slice_selection: 'with_segmentation'  # Only select slices with segmentation
  fig_path: ${general_data_dir}/nifti/figs  # Directory to save the figures

```

#### Optional: Skipping the Step
If the user does not need to perform this quality check, they can skip this step in the pipeline. Simply remove or comment out the following line in the Slurm script:

```bash
python3 evaluation/visualization/nifti_viz.py "${CONFIG_NAME}"

```


