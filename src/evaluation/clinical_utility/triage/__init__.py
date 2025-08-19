# clinical_utility/triage/__init__.py
import importlib
import sys

# ---- Re-export stage entry points ----
from .stage_a import run_stage_a
from .stage_b import run_stage_b
from .stage_c import run_stage_c

# ---- Re-export selected utils at the triage top level ----
from .utils.io import load_csv, make_out_dir, save_table, save_fig
from .utils.preprocessing import prepare_dataframe
from .utils.bootstrap import bootstrap_metric, sens_at_spec
from .utils.plotting import roc_panel, calibration_ax
from .utils.models import train_base_models, mean_ensemble, stack_probs, fit_ensemble
from .utils.plot_style import (
    set_msk_style, cm2inch, pretty, get_joint_colour, get_tissue_style
)
from .utils.demo_flow import DemoCollector, summarise_demographics

# Optional: keep access to the utils namespace if needed
from . import utils as _utils

# ---- Backward-compat submodule aliases (so old sibling imports keep working) ----
# Allow: from .models import ..., etc., inside triage/*
sys.modules[__name__ + ".models"] = importlib.import_module(".utils.models", __package__)
sys.modules[__name__ + ".bootstrap"] = importlib.import_module(".utils.bootstrap", __package__)
sys.modules[__name__ + ".plotting"] = importlib.import_module(".utils.plotting", __package__)
sys.modules[__name__ + ".io"] = importlib.import_module(".utils.io", __package__)
sys.modules[__name__ + ".plot_style"] = importlib.import_module(".utils.plot_style", __package__)
sys.modules[__name__ + ".preprocessing"] = importlib.import_module(".utils.preprocessing", __package__)
sys.modules[__name__ + ".demo_flow"] = importlib.import_module(".utils.demo_flow", __package__)

__all__ = [
    # stages
    "run_stage_a", "run_stage_b", "run_stage_c",
    # utils re-exports
    "load_csv", "make_out_dir", "save_table", "save_fig",
    "prepare_dataframe",
    "bootstrap_metric", "sens_at_spec",
    "roc_panel", "calibration_ax",
    "train_base_models", "mean_ensemble", "stack_probs", "fit_ensemble",
    "set_msk_style", "cm2inch", "pretty", "get_joint_colour", "get_tissue_style",
    "DemoCollector", "summarise_demographics",
    # namespace
    "_utils",
]
