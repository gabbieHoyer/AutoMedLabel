# src/evaluation/statistics/__init__.py

from .utils import remove_brackets_and_convert
from .statistical_tests import (
    compute_shapiro_wilk, compute_levenes_test, compute_spearman_correlation_subject_level
)
from .icc_analysis import (
    perform_icc_analysis, extract_full_icc_info,
    bootstrap_icc_mixed_model, extract_full_icc_info_nonpar
)
from .regression_analysis import compute_regression_results
from .plotting import (
    plot_bland_altman_multiple, plot_regression_comparison,
    plot_gp_regression_subject_level, plot_bland_altman_multiple_nonparametric_subject_level
)
