# ── style_tools.py ────────────────────────────────────────────────
import matplotlib as mpl, seaborn as sns, itertools
from sklearn.metrics import roc_curve, auc as sk_auc

def set_msk_style():
    """defaults: 8 pt DejaVu Sans, thin axes, no frame."""
    sns.set_style("white")
    mpl.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":      8,
        "axes.linewidth": 0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })

# five-hue colour-blind palette
_CB = sns.color_palette("colorblind", 5)
MODEL_COLS = dict(zip(["LR", "RF", "XGB", "Vote", "Stack"], _CB))

# unlimited extra colours if you later plot by visit-set etc.
_CB_EXTRA = sns.color_palette("colorblind", 8)
_colour_cycle = itertools.cycle(_CB_EXTRA)
_label_map: dict[str, tuple[float, float, float]] = {}

def get_colour(label: str):
    """Consistent colour assignment for labels outside MODEL_COLS."""
    if label in MODEL_COLS:
        return MODEL_COLS[label]
    return _label_map.setdefault(label, next(_colour_cycle))

def roc_panel(ax, y_true, y_prob, label, *, auc_ci=None,
              colour=None, linestyle="solid"):
    """Single ROC curve with auto-generated legend entry."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if auc_ci is None:
        auc_val = sk_auc(fpr, tpr)
        leg     = f"{label} (AUC {auc_val:.2f})"
    else:
        auc_val, lo, hi = auc_ci
        leg = f"{label} (AUC {auc_val:.2f} [{lo:.2f}, {hi:.2f}])"

    ax.plot(
        fpr, tpr,
        lw=1.5,
        color=colour or get_colour(label),
        linestyle=linestyle,
        label=leg,
    )

def plot_curve(ax, fpr, tpr, label, *, auc_ci=None,
               colour=None, linestyle="solid"):
    """Plot an already-computed ROC curve (fpr, tpr)."""
    if auc_ci is None:
        auc_val = sk_auc(fpr, tpr)
        leg     = f"{label} (AUC {auc_val:.2f})"
    else:
        auc_val, lo, hi = auc_ci
        leg = f"{label} (AUC {auc_val:.2f} [{lo:.2f}, {hi:.2f}])"

    ax.plot(
        fpr, tpr,
        lw=1.5,
        color=colour,
        linestyle=linestyle,
        label=leg,
    )
