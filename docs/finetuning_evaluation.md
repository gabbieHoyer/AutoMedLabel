# Model Evaluation Scripts (finetune_evaluate.py & finetune_evaluate_biomarker.py)
## Overview
This document provides instructions for running two evaluation pipelines:

1. Dice and IoU Score Evaluation (`finetune_evaluate.py`)
2. Custom Biomarker Metric Evaluation (`finetune_evaluate_biomarker.py`)

Both scripts support multiclass and multi-instance segmentation labels and generate CSV files with results. These CSV tables include metrics at the slice and subject/volume levels.

## Evaluation Pipeline 1: Dice and IoU Score Evaluation
The `finetune_evaluate.py` script computes Dice scores and Intersection over Union (IoU) metrics on the test set for each segmentation class.

### Running the Script
To run the Dice/IoU evaluation, navigate to the root directory and execute the following command:

```bash
$ python src/finetuning/finetune_evaluate.py <yaml config name>
```

Replace <yaml config name> with the name of the YAML configuration file (without the `.yaml` extension). For example, if your configuration file is named dice_eval.yaml, the command would be:

```bash
$ python src/finetuning/finetune_evaluate.py dice_eval
```

### Configuration Files
The configuration files for Dice/IoU evaluation should be located in:

```plaintext
config/finetuning/evaluation
```

### Output Files
The evaluation pipeline generates the following CSV files:

- **Dice and IoU Scores**:
 - **Slice Level**: Metrics for individual image slices.
 - **Subject/Volume Level**: Aggregated metrics per subject or volume.
 - **Global/Dataset Level**: Metrics averaged across the entire dataset.
The output directory for the CSV files is defined in the YAML configuration file.

## Evaluation Pipeline 2: Custom Biomarker Metric Evaluation
The `finetune_evaluate_biomarker.py` script computes custom biomarker metrics based on segmentation masks. It evaluates these biomarkers at both the slice and subject/volume levels and generates separate CSV files for ground truth values and predicted values.

### Running the Script
To run the custom biomarker evaluation, navigate to the root directory and execute the following command:

```bash
$ python src/finetuning/finetune_evaluate_biomarker.py <yaml config name>
```

Replace <yaml config name> with the name of your YAML configuration file (without the `.yaml` extension). For example, if your configuration file is named biomarker_eval.yaml, the command would be:

```bash
$ python src/finetuning/finetune_evaluate_biomarker.py biomarker_eval
```

### Configuration Files
The configuration files for custom biomarker evaluation should be located in:

```plaintext
config/finetuning/evaluation
```

### Output Files
The custom biomarker evaluation pipeline generates the following CSV files:

- **Ground Truth Biomarker CSV**:
 - **Slice Level**: Ground truth biomarker metrics for individual slices.
 - **Subject/Volume Level**: Ground truth biomarker metrics aggregated per subject or volume.

- **Predicted Biomarker CSV**:
 - **Slice Level**: Biomarker metrics derived from the predicted segmentation masks for individual slices.
 - **Subject/Volume Level**: Biomarker metrics derived from predicted masks aggregated per subject or volume.

The output directory for these CSV files is defined in the YAML configuration file.

## YAML Configuration Files
The configuration files for both pipelines must specify various parameters, such as:

- **Dataset**: Information about the test dataset and associated settings.
- **Custom Metrics**: For biomarker evaluation, the configuration file specifies custom metric functions, such as `tissuevolume`, and the relevant tissue classes.
- **Model Weights**: Path to the finetuned model being evaluated.
- **Segmentation Classes**: Definitions for the segmentation classes or instances being evaluated (handled within the dataset configuration).
- **Output Path**: Directory to save the CSV result files.
- **Additional Custom Parameters**: Depending on the evaluation pipeline, additional fields may be required for custom evaluations (e.g., biomarker calculations).

**Example Dice/IoU YAML Config**

```yaml
# ------------------------ Evaluation ------------------------ #
evaluation:
  name: "OAI_T1_Thigh_multiclass_SAM2_mem"
  description: "Evaluating multiclass models finetuned on the OAI thigh muscle MRI dataset."

SEED: 42
distributed: False

dataset:
  OAI-thigh:
    config: examples/OAI_T1_Thigh_plus.yaml
    project: multiclass_Thigh-Evaluation

datamodule:
  max_subject_set: 5
  bbox_shift: 0
  batch_size: 1  
  num_workers: 1

module:
  work_dir: "work_dir/evaluation"
  task_name: multiclass_Thigh-Evaluation
  use_wandb: True
  visualize: True

output_configuration:
  save_path: Run_Summaries
  viz_eval_path: ${output_configuration.save_path}/figs
  summary_file: ${evaluation.name}_data_summary.csv
  logging_level: INFO

base_model: "work_dir/model_weights/SAM2/sam2_hiera_base_plus.pt"

models:
  - finetuned_model: /data/mskprojects/mskSAM/users/ghoyer/AutoMedLabel2/work_dir/finetuning/OAI_Thigh/mskSAM2_mem_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240903-2127_finetuned_model_best_converted.pth
    model_weights: Thigh_multiclass_SAM2_mem-bal_False-sub_set_full-bbox_shift_0
    sam2_model_cfg: sam2_hiera_b+.yaml
    model_details:
      balanced: false
      subject: full
      bbox_shift: 0
    trainable:
      prompt_encoder: false
      image_encoder: true
      mask_decoder: true
```

**Example Biomarker Metric YAML Config**

```yaml
# ------------------------ Evaluation ------------------------ #
evaluation:
  name: "OAI_T1_Thigh_multiclass_biomarker"
  description: "Evaluating models finetuned on MSK MRI dataset for potential in downstream biomarker computation."

SEED: 42
distributed: False

dataset:
  OAI-thigh:
    config: examples/OAI_T1_Thigh_plus.yaml
    project: multiclass_Thigh-Evaluation

datamodule:
  max_subject_set: 5
  bbox_shift: 0
  batch_size: 1
  num_workers: 1
  metric:
    func:
      tissuevolume: true  # Specifies which biomarker metrics to compute
    tissues: [3, 4, 5, 6, 7, 8, 9]  # Defines tissue labels for evaluation
    dicom_fields: ['pixel_spacing', 'slice_thickness', 'rows', 'columns']  # DICOM metadata fields required for computation
    representative_slice: False  # Option for using representative slices for evaluation

module:
  work_dir: "work_dir/evaluation"
  task_name: multiclass_Thigh-Evaluation
  use_wandb: False
  visualize: False

output_configuration:
  save_path: Run_Summaries
  viz_eval_path: ${output_configuration.save_path}/figs
  summary_file: ${evaluation.name}_data_summary.csv
  logging_level: INFO

base_model: "work_dir/model_weights/SAM2/sam2_hiera_base_plus.pt"

models:
  - finetuned_model: /data/mskprojects/mskSAM/users/ghoyer/AutoMedLabel2/work_dir/finetuning/OAI_Thigh/mskSAM2_mem_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240903-2127_finetuned_model_best_converted.pth
    model_weights: Thigh_multiclass_SAM2_mem-bal_False-sub_set_full-bbox_shift_0
    sam2_model_cfg: sam2_hiera_b+.yaml
    model_details:
      balanced: false
      subject: full
      bbox_shift: 0
    trainable:
      prompt_encoder: false
      image_encoder: true
      mask_decoder: true
```

### Key Sections in Biomarker Metric YAML
1. **evaluation**: High-level settings like the experiment name and description.
2. **dataset**: Configuration for the test dataset being evaluated.
3. **datamodule**:
 - **metric.func**: Defines which custom metrics (e.g., `tissuevolume`) are computed.
 - **metric.tissues**: Specifies the list of tissue labels for which the biomarker metrics will be computed.
 - **metric.dicom_fields**: DICOM metadata fields required for computing biomarker metrics like volume or area.
 - **representative_slice**: Option to use a single representative slice for evaluation (useful in some cases).
4. **module**: General configuration related to the evaluation process.
5. **models**: Specifies which finetuned models to evaluate, along with their paths and settings.
6. **output_configuration**: Defines where the evaluation results (CSV files) will be saved, along with visualization paths.

### Adjusting the Configurations
- **Custom Biomarker Metrics**: Ensure that the `metric.func` field includes the biomarker metrics you want to compute, such as `tissuevolume` or others.
- **Tissues**: Set the `metric.tissues` field to match the relevant segmentation labels or tissue types for your evaluation.
- **DICOM Metadata**: Make sure the correct DICOM fields are provided in the `metric.dicom_fields` section, as they are essential for computing metrics like volume.
- **Model Checkpoints**: Ensure the `finetuned_model` path in the YAML file points to the correct finetuned model checkpoint for evaluation.
- **Output Paths**: Set the correct output paths for saving evaluation summaries and visualizations.


## Output Directory Structure
Both evaluation pipelines will save CSV files in directories defined in the YAML config. Be sure to set proper paths to avoid overwriting previous results.

Example directory structure:

```plaintext
output/
│
├── dice_iou_results/
│   ├── slice_dice.csv
│   ├── volume_dice.csv
│   ├── global_dice.csv
│   ├── slice_iou.csv
│   ├── volume_IoU.csv
│   └── global_IoU.csv
│
└── biomarker_results/
    ├── biomarker_gt_slice_volume.csv
    ├── biomarker_gt_volume.csv
    ├── biomarker_pred_slice_volume.csv
    └── biomarker_pred_volume.csv
```

## Advanced Usage
### Custom Metrics (Biomarker Evaluation)
The biomarker evaluation pipeline allows you to define custom metrics that are calculated based on the segmentation masks. These metrics are useful for downstream analysis and clinical applications.


## Troubleshooting
### Common Issues
- **File Not Found**: Ensure all paths to model checkpoints, test datasets, and output directories are correctly set in the YAML configuration.
- **CSV Output Not Generated**: Verify that the model evaluation is completing successfully. Check the logs for any error messages.
- **Multiclass/Multilabel Issues**: Make sure the segmentation classes or labels are properly defined in the dataset config.

## Conclusion
The evaluation scripts (`finetune_evaluate.py` and `finetune_evaluate_biomarker.py`) allow for comprehensive model evaluation using both standard segmentation metrics (Dice/IoU) and custom biomarker metrics. Ensure that the YAML configuration files are properly set up before running the scripts, and use the output CSVs for further analysis.