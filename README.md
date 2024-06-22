## Generalizable Foundation Model for Multi-Tissue Musculoskeletal MRI Segmentation

## mskSAM

![Musculoskeletal MRI Segmentation](assets/mskSAM_butterfly.png)

The focus of this project is on evaluating the potential of foundation models for medical imaging analysis, particularly for musculoskeletal MRI segmentation.

### Motivation

The primary motivation behind this research is to understand the limitations of foundation models trained for the natural imaging domain and assess the challenges in translating these models to complex musculoskeletal anatomy within a rich medical image domain.

### Goals

Our goal is to explore the generalizability of the Segment Anything Model (SAM) when applied to a variety of segmentation tasks common in medical research and clinical settings, focusing on musculoskeletal MRI data.

### Approach

We utilize a diverse collection of musculoskeletal MRI data, assessing SAM's performance in both zero-shot learning and fine-tuning scenarios, to understand its potential for widespread usage in medical imaging pipelines.

## Built With

- Python
- PyTorch

## Dataset

The dataset comprises a wide range of musculoskeletal MR images, spanning various anatomies (knee, spine, hip, thigh), MRI sequences, and quantitative maps, acquired with various fast-acquisition parameters.

## Getting Started

To replicate our research or to apply the foundation model to your musculoskeletal MRI segmentation tasks, follow these setup instructions.

### Prerequisites

Ensure you have Python and PyTorch installed. You will also need other dependencies listed in `requirements.txt`.
```bash
$ pip install -r requirements.txt
```

### Installation and Usage

1. Clone the repository:
```bash
$ git clone https://github.com/gabbieHoyer/mskSAM.git
```

2. Install required packages:
```bash
$ pip install -r requirements.txt
```

3. For detailed instructions on how to run the models and reproduce our results, refer to the `usage.md` document.

## Results

Our findings indicate that SAM, when fine-tuned on a spectrum of musculoskeletal MRI data, shows promising results but requires further evaluation to fully understand its capabilities and limitations in medical imaging analysis.

## Contributing

Contributions to this project are welcome. Please refer to the `CONTRIBUTING.md` for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Department of Radiology and Biomedical Imaging, University of California, San Francisco
- Department of Bioengineering, University of California, Berkeley
- Department of Bioengineering, University of California, San Francisco

## Contact

For any inquiries, please contact:

- Gabrielle Hoyer - gabrielle_hoyer@berkeley.edu
- Michelle Tong - mwtong@berkeley.edu

Project Link: [https://github.com/gabbieHoyer/mskSAM](https://github.com/gabbieHoyer/mskSAM)


