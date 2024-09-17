# Running the Finetuning Script

To run the `finetune_main.py` script, navigate to the root directory of the project and execute:

```bash
$ python src/finetuning/finetune_main.py <config file name>
```

Replace `<config file name>` with the name of your YAML configuration file (without the `.yaml` extension). For example, if the configuration file is named OAI_Thigh_mskSAM2_mem.yaml, the command would be:

```bash
$ python src/finetuning/finetune_main.py OAI_Thigh_mskSAM2_mem
```

#### Example Commands
- Single GPU Execution:

```bash
$ python src/finetuning/finetune_main.py OAI_Thigh_mskSAM2_mem
```

- Distributed Training (SLURM): Ensure that distributed: True is set in the YAML config file for multi-GPU setups.

### Customizing Dataset and Augmentation
The dataset loader and augmentation pipeline can be customized for your specific experiment. You can adjust the augmentation settings, bounding box shift, and number of subjects in the configuration files.

- [Learn more about the Dataset Loader](https://github.com/gabbieHoyer/AutoMedLabel/blob/main/docs/pipeline_details/dataset_dataloader.md)

- [Learn more about the Augmentation Pipeline](path/to/augmentation_pipeline.md)


### Logging and Checkpoints
The script supports:

- **Weights and Biases (WandB) integration**: Set `use_wandb: True` in the configuration to enable logging to WandB.
- **Local Logging**: Configurable with `logging_level` in the config file.
- **Checkpointing**: Intermediate and final model checkpoints are saved in the directory specified by `work_dir` and `save_path` in the config.

### Advanced Features
The finetuning pipeline includes advanced features such as:

- **Early Stopping**
- **Gradient Accumulation**
- **Mixed Precision Training (AMP)**

Example configuration:

```yaml
early_stopping:
  enabled: True
  patience: 10
  min_delta: 0.0001

use_amp: True
grad_accum: 4
```

### More About the Finetuning Engine
To understand how the finetuning engine works, including how loss functions and optimizers are handled, refer to the following documentation:

- [Learn more about the Finetuning Engine](https://github.com/gabbieHoyer/AutoMedLabel/blob/main/docs/pipeline_details/finetuning_engine.md)

### Troubleshooting
- **Distributed Training Issues**: Ensure proper SLURM or multi-GPU configuration.
- **Logging Issues**: If WandB is not working, verify that the API key is set in the environment.

### Conclusion
The `finetune_main.py` script is a powerful tool for fine-tuning SAM2 models. Ensure your configuration files are properly set up and adjust parameters as needed. For more details on dataset loading or augmentation, refer to the additional documentation.