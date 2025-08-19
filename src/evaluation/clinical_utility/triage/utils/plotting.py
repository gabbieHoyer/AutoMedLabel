# triage/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.calibration import calibration_curve

from .plot_style import get_colour

# --------------------------------------------------------------
# Add optional kwargs to roc_panel 
def roc_panel(ax, y, p, label, *, auc_ci=None,
              colour=None, linestyle="solid"):

    fpr, tpr, _ = roc_curve(y, p)
    if auc_ci is None:
        auc_val  = sk_auc(fpr, tpr)
        leg = f"{label} (AUC {auc_val:.2f})"
    else:
        auc_val, lo, hi = auc_ci
        leg = f"{label} (AUC {auc_val:.2f} [{lo:.2f}, {hi:.2f}])"

    ax.plot(
        fpr, tpr,
        lw=1.5,
        color=colour or get_colour(label),
        linestyle=linestyle,
        label=leg,
    )

# palette
main_teal      = '#104A53'
secondary_teal = '#428288'
ref_gray       = '#BBBBBB'
grid_gray      = '#DDDDDD'

def calibration_ax(ax, y, p, title):
    frac, mean = calibration_curve(y, p, n_bins=10)

    # calibration curve in light teal, darker teal marker edges
    ax.plot(
        mean, frac,
        marker='o',
        linestyle='-',
        color=secondary_teal,
        markeredgecolor=main_teal,
        markerfacecolor=secondary_teal,
        linewidth=2,
        markersize=6
    )
    ax.plot([0, 1], [0, 1], "--", color="gray") # diagonal reference in mid-gray
    ax.grid(color=grid_gray, linestyle='-', linewidth=0.5) # light-gray grid
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")


