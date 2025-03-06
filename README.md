# AutoMedLabel

AutoMedLabel is a modular, reproducible pipeline developed to support our study:  
**Scalable Evaluation Framework for Foundation Models in Musculoskeletal MRI Bridging Computational Innovation with Clinical Utility**  
[Preprint on arXiv](https://arxiv.org/abs/2501.13376)

This codebase presents a wide-ranging framework for validating foundation models within a clinical environment, focusing on musculoskeletal MRI. It includes finetuning, evaluation, object detection, and autolabeling, serving as both a reproducible research template and a scalable experimentation platform. The approach centers on determining each model’s capacity to provide benefits in clinical practice, guided by strategies that demonstrate compatibility with existing research and medical workflows while addressing user needs.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/gabbieHoyer/AutoMedLabel.git
    cd AutoMedLabel
    ```

2. **Option 1: Using `requirements.txt`**  
   This approach installs only the dependencies needed to run the code from source—ideal for quick testing.
    ```bash
    pip install -r requirements.txt
    ```
    Then run the pipeline directly from the source using the -m flag:
    ```bash
    python -m src.main [COMMAND] [CONFIG_NAME]
    ```
    *(We recommend this approach rather than python src/main.py to avoid import issues.)*

3. **Option 2: Installing as a Package**  
   Use `setup.py` or `pyproject.toml` to install the project in editable mode for a more polished CLI experience and easier integration.
    ```bash
    pip install -e .
    ```
    With a proper console_scripts entry defined, you can run:
    ```bash
    automedlabel [COMMAND] [CONFIG_NAME]
    ```
    Alternatively, you can still run:
    ```bash
    python -m src.main [COMMAND] [CONFIG_NAME]
    ```
    
*Choose the option that best fits your workflow. For development and quick testing, Option 1 is simple and fast. For a more integrated user experience, Option 2 offers a dedicated command-line entry point.*

## Tested Environments

All code components were tested on Red Hat Enterprise Linux 9.5 with kernel 5.14.0-503.22.1.el9_5.x86_64 and Slurm 23.02.7. Single-GPU tests were performed on various GPUs, including TitanX GPUs (12GB VRAM), which proved sufficient for finetuning, evaluation, autolabel, detection training, and statistical analysis. Multi-GPU training was carried out on HPC nodes featuring Tesla V100 or A100 GPUs (32GB VRAM each) for faster processing, all managed through PyTorch’s Distributed Data Parallel.

## Getting Started with Datasets, Finetuning, and Evaluation Strategies

For detailed instructions on preparing datasets, finetuning models, and evaluating performance, please refer to the documentation in the [docs/](./docs) folder. In particular, check out the [Pipeline Details](./docs/pipeline_details/) section for an in-depth overview of the data preprocessing pipeline. For a concrete example of how to preprocess datasets for scalable use, see the example script: [scripts/build_dataset.sh](./scripts/build_dataset.sh).

Additionally, explore the interactive examples in the [demos/](./demos) folder to see how to run the AutoLabel pipeline, Biomarker analyses, etc.

For detailed study analyses and extended statistical evaluations (including MRI acquisition mixed-modeling and biomarker comparisons), please refer to the `README.md` found in [src/evaluation](./src/evaluation/).

## Usage

After installing AutoMedLabel (see Installation above), run a pipeline by specifying one of the available commands followed by your YAML configuration file name (without the `.yaml` extension).

Available commands include:

- **finetune**: Run finetuning training
- **eval**: Run standard finetuning evaluation
- **eval_biomarker**: Evaluate biomarker-related metrics
- **eval_det2seg**: Evaluate detection-to-segmentation
- **train_det**: Train an object detection model
- **val_det**: Validate an object detection model
- **predict_det**: Predict with an object detection model
- **autolabel**: Run the autolabel pipeline

### Examples

```bash
# Run a finetuning experiment using a config named 'my_experiment.yaml'
python -m src.main finetune my_experiment

# Or, if installed as a package:
automedlabel finetune my_experiment

# Evaluate a model on a config named 'eval_config.yaml'
python -m src.main eval eval_config

# Evaluate detection-to-segmentation on 'eval_det2seg.yaml'
automedlabel eval_det2seg eval_det2seg

# Train an object detection model using 'det_training.yaml'
python -m src.main train_det det_training

# Run the autolabel pipeline in interactive mode using 'Knee.yaml'
python -m src.main autolabel Knee --interactive
```
Each subcommand loads and processes the specified configuration file from your `config/` directory (or wherever you store your `.yaml` configs). Please see the [docs/](./docs) folder for more details on configuring each pipeline.

## License

This project is licensed under the MIT License – see the [`LICENSE`](./LICENSE) file for details.

## Acknowledgments

- **Meta AI** for [Segment Anything](https://github.com/facebookresearch/segment-anything)
- **Ma, J., He, Y., Li, F. et al.** for [MedSAM](https://github.com/bowang-lab/MedSAM/tree/main)
- **Jocher, G., Chaurasia, A., & Qiu, J. (2023)**. *Ultralytics YOLO (Version 8.0.0) [Computer software]*. [GitHub Repository](https://github.com/ultralytics/ultralytics)

## Authors
Gabrielle Hoyer: (https://gabbiehoyer.github.io/)

## Publication
For more details on this work, refer to our preprint:
Scalable Evaluation Framework for Foundation Models in Musculoskeletal MRI Bridging Computational Innovation with Clinical Utility (https://arxiv.org/abs/2501.13376)

## Reference
```bibtex
@misc{hoyer2025scalableevaluationframeworkfoundation,
      title={Scalable Evaluation Framework for Foundation Models in Musculoskeletal MRI Bridging Computational Innovation with Clinical Utility}, 
      author={Gabrielle Hoyer and Michelle W Tong and Rupsa Bhattacharjee and Valentina Pedoia and Sharmila Majumdar},
      year={2025},
      eprint={2501.13376},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.13376}, 
}
```


