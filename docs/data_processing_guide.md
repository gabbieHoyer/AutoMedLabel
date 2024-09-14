## Data Processing Guide for mskSAM
This document provides detailed instructions on preparing your musculoskeletal MRI datasets for segmentation with the mskSAM project. It covers the steps to standardize medical images, create metadata, and prepare data for the Segment Anything Model (SAM).

### Overview
The `data_processing` module in the mskSAM project includes several scripts designed to process musculoskeletal MRI data for use with machine learning models. This guide focuses on three primary scripts:

- `data_standardization.py`: Converts medical images and segmentation masks to NIfTI format.
- `sam_prep.py`: Converts images to a 2D slice `.npy` format compatible with the SAM model.
- `slice_standardization.py`: Generates three forms of metadata for each dataset.

Each step requires specific configuration files for the datasets you are working with. These .yaml files should be located in the `root/config` directory.

## Default Outputs
```
.                                       # Dataset folder
├── ...
├── nifti                               # Standardized 3D volume data in NIfTI format
│   ├── imgs                            # Images
│   │   ├── volume_id.nii.gz               
│   │   └── ...                            
│   ├── masks                           # Mask segmentations 
│   │   ├── volume_id.nii.gz               
│   │   └── ...                              
│   └── figs                            # Figures for quality control 
│       └── ...                              
├── npy                                 # 2D slice data in NPY format
│   ├── imgs                            # Images. 
│   │   ├── volume_id-###.npy           # "###" refers to slice number with leading zeros.
│   │   └── ...                            
│   ├── masks                           # Mask segmentations 
│   │   ├── volume_id-###.npy           
│   │   └── ...                         
│   └── figs                            # Figures for quality control 
│       └── ...                         
├── metadata                            # Info about each image volume and mask
│   ├── metadata_for_stats.json         # Info about dataset, volume demographics, and slice dicom header 
│   ├── metadata_for_ml.json            # Info about split and only metadata relevant to ML (sex, age, ...)
│   └── slice_paths                     # Path info about each set of (imgs, masks) NPY file 
│       ├── volume_id-###.parquet       #     (same file names as NPY)
│       └── ...                            
└── ...
```

### Prerequisites
Before you begin, ensure you have completed the installation steps outlined in the project's README. You will need Python, PyTorch, and other dependencies installed. Also, ensure you have the `{dataset_name}.yaml` configuration files for your datasets ready in the `config` directory. 

### Step 1: Data Standardization
1. Navigate to the `src/data_processing` directory.
2. Run `data_standardization.py` with the appropriate `.yaml` config file. This script will process your MRI images and segmentation masks, converting them into the NIfTI format for further processing.

    ```bash
    $ python data_standardization.py {dataset_name}
    ```

Replace {dataset_name} with the name of your dataset without the .yaml extension.

3. (Optional) Run `nifti_visualization.py` with the appropriate `.yaml` config file. This script save figures of nifti images and overlaid segmentation masks. It is possible to save a gif or 2D image.

    ```bash
    $ python nifti_visualization.py {dataset_name}
    ```

### Step 2: Preparing Data for SAM
1. To prepare your data for the Segment Anything Model (SAM), use the `data_sam_prep.py` script. This will convert your images into a 2D slice 
`.npy` format that is compatible with SAM.

    ```bash
    $ python sam_prep.py {dataset_name}
    ```

### Step 3: Metadata Creation
1. After standardizing your data, you can create the necessary metadata files using `metadata_creation.py`. This script generates three forms of metadata, each essential for different aspects of the project. You must specify the operation type (`--operation`) when running the script, depending on the metadata you need to generate.

Pre-requisites
For operation A, ensure you have the original DICOM image data available as it extracts DICOM header information for statistics and documentation.
For operations B and C, ensure that files have been processed into the volume (NIfTI) format and converted to `.npy` slice format.

Generating Metadata
Run `metadata_creation.py` from the `src/data_processing` directory, specifying the configuration file and the operation you wish to perform:

```bash
$ python metadata_creation.py {dataset_name} --operation [A|B|C]
```

Replace `{dataset_name}` with the name of your dataset without the `.yaml` extension and choose the operation (`A`, `B`, or `C`) based on the type of metadata you wish to create:

- **Operation A**: For statistics metadata

    ```bash
    $ python metadata_creation.py {dataset_name} --operation A
    ```

This operation is suitable when you need to gather statistical information from the original DICOM images. It's useful for initial data analysis and documentation.

- **Operation B**: For primary metadata with splits

    ```bash
    $ python metadata_creation.py {dataset_name} --operation B
    ```

Use this operation to create primary metadata that includes data splits (e.g., training, validation, testing). It requires data in NIfTI format and is crucial for setting up your dataset for model training and evaluation.

- **Operation C**: For subject slice metadata

    ```bash
    $ python metadata_creation.py {dataset_name} --operation C
    ```

This operation is necessary for preparing the dataset for segmentation tasks with the SAM model, converting volume data into a slice-based `.npy` format for each subject.

Ensure that your configuration files in the config directory are correctly set up with paths to your datasets and other necessary parameters before running these commands.

### Additional Information
Ensure that your `.yaml` configuration files are correctly set up with paths to your datasets and other necessary parameters.
For detailed explanations of configuration options and additional parameters for each script, refer to the comments within the script files themselves.

### Support
For any questions or issues encountered during the data processing steps, please refer to the project's issue tracker or contact the contributors listed in the README :D