# NIfTI to NPY Slice Standardization for SAM/SAM2 Fine-Tuning and YOLO-Compatible Output

The Slice Standardization step processes MRI images and corresponding segmentation masks into .npy format, enabling the preparation of data for fine-tuning segmentation models (such as SAM/SAM2) and generating YOLO-compatible annotations for object detection models. This step involves resizing, normalizing, and saving individual slices of the 3D image volumes and masks, ensuring compatibility for training and fine-tuning.

### Purpose
- **SAM/SAM2 Fine-Tuning**: Preprocesses MRI slices into a format compatible with SAM and SAM2, facilitating the fine-tuning process.
- **YOLO-Compatible Output (Optional)**: Prepares `.npy` files for MRI image slices and generates bounding box annotations for YOLO-style object detection if specified.
- **Slice-Level Data Preparation**: Converts 3D MRI image volumes and segmentation masks into 2D slices, saving them individually for model training.

#### Important Note:
- For SAM models, both the image and mask files must be 1024x1024.
- For SAM2, the image files can be set to 1024x1024 to maintain resolution and fine features, while the mask files can be processed at 256x256, allowing for faster training and larger batch sizes.

### Key Components
#### Script: `slice_standardization.py`
This script converts NIfTI format data into .npy files, preparing the MRI images and masks for SAM/SAM2 and YOLO-compatible processing. It handles operations such as image resizing, cropping non-zero slices, and generating bounding boxes for YOLO-style object detection.

### Steps Performed
1. Loading Configurations: The script starts by loading a dataset-specific YAML configuration file. It retrieves paths to the NIfTI images and masks and sets parameters for how the data should be processed (e.g., image size, cropping, voxel thresholds).

2. Mask and Image Preparation:
- **Mask Preparation (`MaskPrep` Class):**
  - Removes unwanted labels from the mask (e.g., background or specific structures not relevant for training).
  - Labels connected components as instances for certain masks (e.g., individual bones or tissues).
  - Removes small objects below a specified voxel or pixel threshold to clean up the masks.
  - Resizes the mask to the desired output size while preserving label integrity using nearest-neighbor interpolation.
  - Optionally crops the mask to include only slices with non-zero values.
- **Image Preparation (`ImagePrep` Class):**
  - Clips outlier intensities from the image data based on percentile ranges.
  - Normalizes the image intensity values to a range of [0, 255] for SAM processing.
  - Resizes the images to the target size using cubic spline interpolation.

3. **Slice-Level Processing**: The script processes each 3D NIfTI volume slice by slice:
- **Images**: Each slice is resized, normalized, and saved as a 3-channel `.npy` file in the `imgs/` directory.
- **Masks**: Each slice of the segmentation mask is preprocessed and saved as a single-channel `.npy` file in the `gts_256/` directory (for SAM2, masks are saved at 256x256 resolution).

4. YOLO-Compatible Output (Optional): If yolo_compatible: True is set in the configuration, the script generates bounding boxes for each instance within the mask. The bounding boxes are saved in `.txt` files, which are stored in the `labels_256/` directory, formatted for YOLO-style object detection models.

5. Saving Processed Data: The processed image and mask slices are saved in separate directories. Each slice is saved with a consistent naming convention that includes the base name and the slice index (e.g., `base_name-###.npy`).

### Configuration Parameters (YAML)
The slice standardization step relies on the YAML configuration file to determine how the data should be processed. Key parameters include:

| Parameter            | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| `nifti_image_dir`     | Path to the directory containing the NIfTI images.                       |
| `nifti_mask_dir`      | Path to the directory containing the NIfTI masks.                        |
| `npy_dir`             | Output directory for the processed `.npy` files.                         |
| `preprocessing_cfg`   | Configuration for image and mask preprocessing (resize, cropping, thresholds). |
| `image_size`          | Target size for resizing images and masks.                               |
| `voxel_num_thre2d`    | Minimum voxel count threshold for 2D slices (to remove small objects).   |
| `voxel_num_thre3d`    | Minimum voxel count threshold for 3D volumes (to remove small objects).  |
| `remove_label_ids`    | List of label IDs to be removed from the masks.                          |
| `target_label_id`     | Specific label IDs to target for instance segmentation (e.g., individual tissues). |
| `yolo_compatible`     | Whether to generate YOLO-compatible data, including bounding boxes.      |


### Example Workflow
1. **Running Slice Standardization**: To process NIfTI images and masks into .npy format for SAM/SAM2 fine-tuning or YOLO-compatible output, run the following command:

```bash
python3 preprocessing/slice_standardization.py OAI_thigh_muscle
```

This command will convert the 3D NIfTI volumes into 2D .npy slices, resizing the images and masks as specified in the configuration file.

2. **Configuration for SAM/SAM2**: Example YAML configuration for preparing data for SAM/SAM2 fine-tuning:

```yaml
preprocessing_cfg:
  image_size: 1024
  voxel_num_thre2d: 100
  voxel_num_thre3d: 200
  remove_label_ids: [2]  # Remove fascia label in this case
  target_label_id: null  # No specific label targeted for instance segmentation
  instance_bbox: True
  make_square: True
  ratio_resize: False
  yolo_compatible: False  # Set to True if YOLO-compatible output is needed

```

3. **Configuration for YOLO-Compatible Output**: If YOLO-compatible output is needed, the script will additionally generate bounding box annotations for the objects of interest in each slice:

```yaml
preprocessing_cfg:
  image_size: 1024
  voxel_num_thre2d: 100
  voxel_num_thre3d: 200
  remove_label_ids: [2]
  target_label_id: null
  instance_bbox: True
  make_square: True
  ratio_resize: False
  yolo_compatible: True  # Enable YOLO-compatible output

```

The script will output the `.npy` files for images and masks, as well as the bounding box annotations in `.txt` files for each slice.

#### Optional: Skipping YOLO-Compatible Output
If YOLO-compatible output is not required, set yolo_compatible: False in the YAML configuration. This will skip the bounding box generation step, focusing only on the preparation of data for SAM/SAM2 fine-tuning.


