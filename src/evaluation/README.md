## Study Analyses and Extended Statistical Evaluations

The `src/evaluation/` directory houses the scripts used in our study. These analyses include:

- **Biomarker Analysis:**  
  The `biomarker_analysis.py` script compares clinically relevant biomarker metrics (e.g., cartilage thickness, muscle volume, T1ρ, etc.) derived from manual segmentations with those obtained via finetuned model inference. It uses statistical tests (ICC analysis, Bland–Altman plots, regression comparisons) to assess whether the automated segmentation approach can reliably substitute for manual annotation in downstream clinical tasks.

- **MRI Acquisition Analysis:**  
  The `mri_acquisition_analysis.py` script processes raw MRI acquisition data through data cleaning, feature selection, and mixed-effects modeling. By fitting models that include key imaging parameters (e.g., TE, pixel spacing, slice thickness, flip angle) and their interactions, this analysis evaluates how variability in MRI acquisition affects model performance when different training strategies (such as dataset size, model freezing, prompt augmentation, and dataset mixing) are employed.

An interactive demo for the biomarker analysis is available in the [demos/](../../demos) folder.
