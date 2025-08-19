# plot_style.py
import matplotlib as mpl
import seaborn as sns
import itertools

def set_msk_style():
    """
    Apply defaults:
      • DejaVu Sans 8 pt
      • thin axes lines, no right/top spines
      • no legend frame
      • seaborn white background
    """
    sns.set_style("white")
    mpl.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":      8,
        "axes.linewidth": 0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })

def cm2inch(*dims_cm):
    return tuple(d / 2.54 for d in dims_cm)

# five-colour colour-blind palette (matches panel f)
CB_COLS = sns.color_palette("colorblind", 5)

# map your five models to those same hues
MODEL_COLS = dict(zip(
    ["LR", "XGB", "HGB", "ENS", "STACK"],
    CB_COLS
))

# ------------------------------------------------------------------
#  Fallback colour cycle for any other labels (stage-C joints, etc.)
# ------------------------------------------------------------------
CB_EXTRA = sns.color_palette("colorblind", 8)
_colour_cycle = itertools.cycle(CB_EXTRA)   # never runs out
_label_map: dict[str, tuple[float, float, float]] = {}

def get_colour(label: str):
    if label in MODEL_COLS:
        return MODEL_COLS[label]

    # assign a colour the first time we see this label
    return _label_map.setdefault(label, next(_colour_cycle))

# ------------------------------------------------------------------
#  Pretty display names for Stage-C labels
# ------------------------------------------------------------------
_PRETTY = {
    # p1 anomaly tasks
    "femur_anom":   "Femoral Abnormality",
    "tibia_anom":   "Tibial Abnormality",
    "patella_anom": "Patellar Abnormality",

    # p2 cartilage
    "femur_cart":   "Femoral Cartilage",
    "tibia_cart":   "Tibial Cartilage",
    "patella_cart": "Patellar Cartilage",

    # p2 bone
    "femur_bone":   "Femur",
    "tibia_bone":   "Tibia",
    "patella_bone": "Patella",
}

def pretty(label: str) -> str:
    """Return a cleaner display string."""
    return _PRETTY.get(label, label.replace("_", " ").title())

# --------------------------------------------------------------
# In plot_style.py – add two dictionaries
JOINT_HUES = {
    "femur":   "#93C3CA",   # pale aqua
    "tibia":   "#1A5063",   # deep teal-blue
    "patella": "#954966",   # muted plum
}

TISSUE_LSTY = {         # will be checked in plotting
    "cart":  "solid",  #(0, (5, 3)),   # dashed
    "bone":  "solid",
}

# helper
def get_joint_colour(label):
    # label is e.g. "femur_cart" or "femur_bone"
    joint = label.split("_")[0]
    return JOINT_HUES[joint]

def get_tissue_style(label):
    tissue = label.split("_")[1]          # cart / bone
    return TISSUE_LSTY[tissue]
