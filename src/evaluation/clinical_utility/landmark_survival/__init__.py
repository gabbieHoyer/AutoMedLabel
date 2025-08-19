# clinical_utility/landmark_survival/__init__.py
import importlib
import sys

# ---- Re-export stage-facing API (utils surfaced at package top level) ----
from .utils.data_utils import (
    make_feature_matrix_demo_only,
    get_dc_scenarios,
    get_feature_fn,
    get_event_time_col,
    BIOMETRIC_COLS,
    DEMOGRAPHIC_COLS,
    TIMEPOINTS_ALL,
    clean_oai,
)
from .utils.model_utils import get_lr, get_rf
from .utils.outcomes import get_label_fn
from .utils.shap_plots import save_shap_bar, save_shap_beeswarm, save_dependence_grid
from .utils.style_tools import set_msk_style, roc_panel, get_colour

# Optional: expose utils namespace
from . import utils as _utils

# ---- Back-compat submodule aliases (keep old imports working) ----
# Allow: from data_utils import ..., etc., inside landmark_survival/*
pkg = __package__
sys.modules[__name__ + ".data_utils"] = importlib.import_module(".utils.data_utils", pkg)
sys.modules[__name__ + ".model_utils"] = importlib.import_module(".utils.model_utils", pkg)
sys.modules[__name__ + ".outcomes"] = importlib.import_module(".utils.outcomes", pkg)
sys.modules[__name__ + ".shap_plots"] = importlib.import_module(".utils.shap_plots", pkg)
sys.modules[__name__ + ".style_tools"] = importlib.import_module(".utils.style_tools", pkg)

__all__ = [
    # data utils
    "make_feature_matrix_demo_only",
    "get_dc_scenarios",
    "get_feature_fn",
    "get_event_time_col",
    "BIOMETRIC_COLS",
    "DEMOGRAPHIC_COLS",
    "TIMEPOINTS_ALL",
    "clean_oai",
    # models
    "get_lr",
    "get_rf",
    # labels
    "get_label_fn",
    # SHAP figs
    "save_shap_bar",
    "save_shap_beeswarm",
    "save_dependence_grid",
    # style
    "set_msk_style",
    "roc_panel",
    "get_colour",
    # namespace
    "_utils",
]
