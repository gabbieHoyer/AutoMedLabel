# Finetuning and Evaluation Engine for SAM2 Models

## Overview
The engine is the core component responsible for managing the training and evaluation loops for SAM2 models in the finetuning pipeline. It allows you to:

- Finetune SAM2 models with customized training, optimization, and scheduling strategies.
- Evaluate models using metrics such as Dice scores and IoU.
- Perform evaluations with custom biomarker metrics.
- The engine is designed to be flexible, supporting multi-GPU distributed training, early stopping, gradient accumulation, and mixed precision (AMP) training. It also includes visualization tools for model predictions and results.

This document describes how the engine works for three main scenarios:

1. **Finetuning SAM2 models**
2. **Evaluating with Dice/IoU metrics**
3. **Evaluating with custom biomarker metrics**

## Engine Components
### 1. Model Initialization and Finetuning
The engine starts by setting up the model using the provided SAM2 configurations and initial weights (pretrained or checkpointed models). The main steps include:

- Loading the SAM2 model with a predefined configuration (`sam2_model_cfg`).
- Preparing the finetuning model, including selecting trainable components (e.g., image encoder, mask decoder).
- Setting up the optimizer and learning rate scheduler for the training process.
- Optionally resuming from a checkpoint if available.

**Example: Model Initialization for Finetuning**
The model is initialized with a configuration that specifies which parts to finetune:

```python
sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
finetuned_model = finetunedSAM2(model=sam2_model, config=comp_cfg).to(device)
```

The `finetunedSAM2` model allows for selective finetuning of SAM2 components, such as the image encoder and mask decoder.

#### Model Architecture Customization
The finetuning pipeline provides the flexibility to alter different components of the SAM2 model architecture based on the experiment configuration. The SAM2 model class allows you to select which parts of the architecture should be trainable or frozen during the finetuning process. This includes:

- **Image Encoder**
- **Mask Decoder**
- **Prompt Encoder**

You can configure which components are trainable via the configuration file, and the model will adjust accordingly during the finetuning process. For example, you can choose to freeze the image encoder while keeping the mask decoder trainable.

#### Location of SAM2 Model Class
The SAM2 model class, which enables these architectural customizations, can be found in:

```bash
src/finetuning/engine/models/sam2.py
```

This class is responsible for setting up the SAM2 model and controlling which parts of the model are trainable during finetuning. Here’s an excerpt of the class showing how components can be frozen:

```python
class finetunedSAM2(nn.Module):
    def __init__(self, model, config):
        super().__init__()

        self.sam2_model = model

        # Apply trainable configuration / Freeze components based on config
        if not config['prompt_encoder']:
            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

        if not config['image_encoder']:
            for name, param in self.sam2_model.named_parameters():
                param.requires_grad = False

            for name, param in self.sam2_model.named_parameters():
                if 'sam_mask_decoder' in name:
                    param.requires_grad = True
```
In the configuration file, you can specify which components should be trainable. For example:

```yaml
module:
  trainable:
    prompt_encoder: false
    image_encoder: true
    mask_decoder: true
```
In this configuration, the **prompt encoder** will be frozen, while the **image encoder** and **mask decoder** will be trainable during the finetuning process.

### 2. Training Process
The Trainer class manages the training loop for finetuning SAM2 models. Key features of the training process include:

- **Loss Calculation**: The engine computes the loss as a combination of Dice loss and binary cross-entropy (BCE) loss, with configurable weighting.
- **Gradient Accumulation**: If large batches don't fit into GPU memory, the engine supports accumulating gradients over smaller batches.
- **Mixed Precision Training (AMP)**: For memory efficiency, the engine can perform training with automatic mixed precision (AMP), using a gradient scaler when necessary.
- **Early Stopping**: The engine monitors validation loss and automatically halts training if the loss does not improve after a certain number of epochs (patience).

#### Loss Calculation
The loss is computed for each batch using a combination of segmentation (Dice) loss and cross-entropy loss:

```python
def calculate_loss(self, predictions, gt2D, seg_loss_weight=0.5, ce_loss_weight=0.5):
    seg_loss, ce_loss = self.loss_fn
    total_loss = seg_loss_weight * seg_loss(predictions, gt2D) + ce_loss_weight * ce_loss(predictions, gt2D.float())
    return total_loss
```

This combined loss helps optimize the model for both segmentation quality and accuracy.

#### Early Stopping
If the validation loss does not improve after a certain number of epochs, early stopping is triggered:

```python
def check_early_stopping(self, val_loss):
    if val_loss + self.min_delta < self.best_val_loss:
        self.no_improve_epochs = 0
    else:
        self.no_improve_epochs += 1
    if self.no_improve_epochs >= self.patience:
        self.early_stopped = True
```

### 3. Optimizer and Scheduler Customization
One of the key strengths of the engine is its flexibility to swap out different optimizers and learning rate schedulers based on the experiment requirements.

In the configuration file (`.yaml`), you can specify different optimizers and schedulers with their respective parameters, making it easy to adapt the training process for different models or tasks.

**Example: Optimizer Configuration**
In the **experiment config file**, you can specify parameters for the optimizer like this:

```yaml
module:
  optimizer:
    type: "AdamW"  # You can replace this with SGD, Adam, RMSprop, etc.
    lr: 0.0001
    weight_decay: 0.01
```

By default, the engine uses `AdamW` for optimization, but you can switch to another optimizer like `SGD` or `Adam` by specifying the type. This allows you to experiment with different training strategies without modifying the core engine code.

**Example: Scheduler Configuration**
Similarly, the learning rate scheduler can be configured in the config file:

```yaml
module:
  scheduler:
    type: "CosineAnnealingWarmRestarts"  # or CosineAnnealingLR, StepLR, etc.
    eta_min: 0.00001
    T_max: 10
```

This configuration enables the engine to use a `CosineAnnealingWarmRestarts` scheduler, but you can replace this with any PyTorch-compatible scheduler, such as `StepLR` or `ExponentialLR`.


### 4. Evaluation with Dice/IoU Metrics
When evaluating a finetuned model, the engine computes segmentation metrics like Dice scores and IoU for multiclass datasets. This process is similar to training but without the need for backpropagation.

Key features:

- **Multi-class Segmentation**: The engine can evaluate models on datasets with multiple segmentation classes.
- **Bounding Box Prompts**: For SAM2 models, bounding boxes are computed for each ground truth mask instance, and these are used to prompt the model to predict segmentation masks.

#### Dice Metric Calculation
The **Dice metric** is computed for each predicted mask and ground truth:

```python
self.dice_metric(y_pred=predictions_binary, y=gt2D)
```

At the end of each validation epoch, the Dice score is aggregated across all batches:

```python
dice_score_tensor = self.dice_metric.aggregate()
```

This ensures a consistent and robust evaluation across multiple images and instances.


### 5. Logging and Visualization
The engine supports various logging and visualization features:

- **Weights and Biases (WandB)**: The engine can log experiment details and metrics to WandB for easy tracking and comparison of results.
- **Visualization**: During the finetuning process, the engine generates visualizations of model predictions, showing both the ground truth and predicted masks. These visualizations are saved in the output directory.

**Visualization Example**
The `visualize_input` function is used to generate and save visualizations of the model's predictions for debugging and quality control:

```python
visualize_input(
    image=image_for_viz[i],                  
    gt_mask=gt_mask_for_viz[i].squeeze(),  
    box=box_for_viz[i],  
    pred_mask=pred_mask_for_viz[i].squeeze(),
    label_id=label_id[i].item(),  
    image_name=f"{img_name[i]}_QC",
    model_save_path=self.model_save_path
)
```
This provides a clear visual comparison between the model's predicted segmentation and the ground truth.


## Evaluation with Dice/IoU Metrics
The evaluation pipeline provides a flexible way to compute **Dice and Intersection over Union (IoU)** metrics during model evaluation. The engine enables evaluations on both multi-label and single-label datasets, handling multiple instances of segmentation classes and complex data configurations.

Similar to the finetuning engine, the evaluation engine supports features like **distributed processing, visualization**, and logging to **Weights & Biases (W&B)**.

#### Key Features of the Evaluation Engine
- **Multilabel and Single-label Evaluation:**
 - For multilabel datasets, the engine generates predictions for each segmentation label and its instances using bounding box prompts.
 - For single-label datasets, it computes predictions across a variety of segmentation labels without the need for instance-based prompting.

- **Dice and IoU Computation:**
 - Dice and IoU metrics are computed on both the slice level and the subject/volume level.
 - These metrics are logged for each segmentation class, providing both global and per-class scores.

- **Instance Handling:**
 - The evaluation pipeline is capable of handling multiple instances of each class, creating a bounding box for each instance and evaluating predictions at this fine-grained level.

- **Custom Visualizations:**
 - Visualizations of predictions can be generated for sampled datasets during evaluation. This allows for **quality checks** to ensure that the model’s predictions align with the ground truth.


## Engine Class for Dice/IoU Evaluation
The evaluation process for **Dice** and **IoU** is implemented in the `Tester` class. This class is responsible for:

- Loading the trained model and preparing the test dataset.
- Iterating over the test set, generating predictions, and calculating the **Dice** and **IoU** scores for each batch.
- Handling multilabel and single-label datasets, with support for multiple instances of segmentation labels.

Below is a simplified overview of the `Tester` class responsible for evaluation:

```python
class Tester:
    def __init__(self, model, test_loader, eval_cfg, module_cfg, datamodule_cfg, experiment_cfg, run_path, device='cpu', data_type='full', visualize=False):
        # Model and data loaders setup
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.visualize = visualize
        
        # Metrics initialization (Dice and IoU)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", num_classes=datamodule_cfg['num_classes'])
        self.IoU_metric = MeanIoU(include_background=False, reduction="mean_batch")

    def evaluate_multilabel_test_set(self):
        # Loop through the test set, computing metrics for multilabel datasets
        for batch_idx, batch in enumerate(self.test_loader):
            images, gt2D, boxes, label_ids = batch['image'], batch['gt2D'], batch['boxes'], batch['label_ids']
            predictions = self.model(images, boxes)  # Generate predictions for each bounding box
            
            # Calculate Dice and IoU for the predictions
            dice_score = self.dice_metric(predictions, gt2D)
            IoU_score = self.IoU_metric(predictions, gt2D)

    def save_metric_scores_to_csv(self, metric_name, slice_metric):
        # Save per-slice and per-volume metrics to CSV files
        df = pd.DataFrame(slice_metric)
        df.to_csv(f"{self.run_path}/slice_{metric_name}.csv", index=False)
```

### Dataset-specific Considerations
For multilabel datasets, the engine handles multiple segmentation instances by:

- Generating bounding boxes for each instance.
- Computing **Dice** and **IoU** for each instance of each class.
In single-label datasets, the engine handles the entire segmentation mask at once, without instance-specific processing.

### Logging and Saving Results
During evaluation, metrics are saved to CSV files at three levels:

1. **Slice Level**: Per-slice Dice and IoU scores for each class.
2. **Volume/Subject Level**: Aggregated scores across slices for each subject.
3. **Global Level**: Global Dice and IoU scores across the entire test set.

The engine also supports logging metrics to **Weights & Biases (W&B)** for tracking and visualization.

### Swapping Metrics or Adding Custom Metrics
Similar to the finetuning engine, you can modify or extend the metrics used for evaluation. The **Dice** and **IoU** metrics are defined in the experiment configuration file and can be replaced or augmented with additional metrics. For example, you could introduce **precision**, **recall**, or other relevant metrics by adding them to the evaluation logic.


## Evaluation with Custom Biomarker Metrics
The **SAM2 finetuning evaluation pipeline** provides flexibility for evaluating model performance not only based on traditional segmentation metrics such as **Dice** and **IoU**, but also through custom biomarker metrics. These biomarkers can be integral to clinical analysis, assessing tissue properties or other medical characteristics such as **T1rho**, **T2 maps**, and **cartilage thickness**.

### Key Features of the Custom Biomarker Evaluation Engine
- **Biomarker Metrics**: In addition to traditional segmentation metrics, custom metrics like **cartilage thickness**, **tissue height**, **tissue volume**, and **T1rho/T2 maps **can be evaluated. These metrics provide clinical insights into the health of tissues in the segmentation masks, such as evaluating degenerative conditions in cartilage.

- **Metric Factory for Custom Metrics**: The engine relies on a dynamic **metric factory** to load and compute these custom metrics. This design allows for easy integration of new metrics and ensures flexibility based on dataset-specific needs.

### Biomarker Configuration in Experiment Setup
To configure the evaluation with custom biomarker metrics, you need to specify the relevant metrics, tissues, and metadata in the experiment's YAML configuration. Here's an example:

```yaml
datamodule:
  max_subject_set: 3
  bbox_shift: 0
  batch_size: 1  
  num_workers: 1
  metric:
    func:
      cartilagethickness: true
      t1rho: true
    tissues: [1, 2, 3, 4]  # Specify tissues to evaluate
    dicom_fields: ['pixel_spacing', 'slice_thickness', 'rows', 'columns']
    representative_slice: True  # Whether to evaluate based on a representative slice
```
This configuration directs the engine to use the `cartilage thickness` and `T1rho` metrics for the specified tissues, incorporating DICOM metadata fields like pixel spacing, slice thickness, rows, and columns.

- `representative_slice: True` indicates that evaluation will utilize metadata from a singular representative MRI slice for the subject. The biomarker metrics (such as cartilage thickness) will be computed based on this slice, and these values are extracted from the `metadata_for_ml.json` file. This approach assumes that the chosen slice represents the entire subject.

- `representative_slice: False` can be set when slice-level MRI metadata is required for more granular evaluations. In this case, the `metadata_for_stats.json` file is used, providing detailed slice-by-slice MRI information to compute biomarker metrics for the entire subject volume.

This flexibility allows the system to adjust the granularity of the evaluation based on the needs of the specific biomarker or study.

### How the Metric Factory Works
The custom biomarker metrics are loaded via the metric factory located in `src/finetuning/engine/metrics/metric_factory.py`. This factory dynamically selects which metric class to instantiate based on the configuration:

```python
def metric_factory(metric_name, **kwargs):
    if metric_name.lower() == 't1rho':
        return T1rhoMetric(**kwargs)
    elif metric_name.lower() == 't2':
        return T2Metric(**kwargs)
    elif metric_name.lower() == 'tissueheight':
        return TissueHeightMetric(**kwargs)
    elif metric_name.lower() == 'tissuevolume':
        return TissueVolumeMetric(**kwargs)
    elif metric_name.lower() == 'cartilagethickness':
        return CartilageThicknessMetric(**kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_name}")
```

The factory determines which metric to use based on the name specified in the YAML configuration under `metric.func`. For instance, if **cartilage thickness** is required, the factory will return an instance of `CartilageThicknessMetric`.

### Loading Multiple Metrics
The engine loads multiple metrics using the `load_metrics` function, which processes the configuration and initializes each metric with the required parameters. The `class_names` and `tissue_labels` are passed to ensure that metrics are computed for the correct classes or tissue types:

```python
def load_metrics(config, class_names=None, tissue_labels=None):
    metrics = []
    metric_params = {
        'reduction': 'mean_batch',
        'class_names': class_names,
        'tissue_labels': tissue_labels
    }

    for metric_name, use_metric in config.items():
        if use_metric:
            metrics.append(metric_factory(metric_name, **metric_params))

    return metrics
```

### Biomarker Evaluation Engine: finetune_engine_eval_biomarker.py
The engine for biomarker evaluation (`finetune_engine_eval_biomarker.py`) is responsible for computing the custom metrics and logging the results. It uses the loaded metrics and processes each batch of test data, evaluating segmentation performance alongside the specified biomarker metrics.

```python
class Tester:
    def __init__(self, model, test_loader, eval_cfg, module_cfg, datamodule_cfg, experiment_cfg, run_path, device='cpu', data_type='full', visualize=False):
        self.metrics = load_metrics(self.datamodule_cfg['metric']['func'], self.class_names, self.tissue_labels)
        self.metadata_dict = load_meta(self.datamodule_cfg['stats_metadata_file'], self.representative_slice)
    
    def evaluate_multilabel_test_set(self):
        for batch in self.test_loader:
            for metric in self.metrics:
                t1rho, t2 = batch.get('t1rho'), batch.get('t2')
                subj_id, slice_id = parse_image_name(batch['img_name'])
                subject_slice_meta = extract_meta(self.metadata_dict, subj_id, self.dicom_fields, slice_id=slice_id)
                
                # Apply the metric to predictions and ground truth
                metric.update(y_pred=predictions, y=gt2D, map_array=t1rho if 't1rho' in metric_name else t2, subject_slice_meta=subject_slice_meta)
    
    def save_custom_metric_scores(self, metric_name, total_volumes_pred, total_volumes_true, aggregated_volumes_pred, aggregated_volumes_true):
        # Save the computed metrics to CSV or log them
```

The **biomarker evaluation engine** in SAM2 provides a highly flexible and modular way to assess both segmentation and clinical biomarkers. By using a **metric factory**, custom metrics such as **cartilage thickness** and **T1rho/T2** maps can be seamlessly integrated into the evaluation process, providing deeper insights into model performance and clinical relevance.

The biomarker evaluation engine is located in `finetune_engine_eval_biomarker.py`, and its configuration can be easily extended through YAML files. This flexibility ensures that SAM2 can be tailored to a wide range of clinical tasks, making it a powerful tool for medical imaging research.

## Conclusion
The SAM2 finetuning and evaluation pipeline is a highly flexible and modular system designed to handle a wide variety of tasks in medical image segmentation. Its versatile architecture allows for customizations at every stage, from finetuning different model components to evaluating segmentation performance with traditional metrics like Dice and IoU, as well as custom clinical biomarker metrics such as cartilage thickness and T1rho/T2 maps. With configurable optimizers, schedulers, and a metric factory for dynamic metric selection, this system is equipped to meet diverse research and clinical needs. The design ensures adaptability, making it a powerful tool for fine-tuning and evaluating models in medical imaging applications.