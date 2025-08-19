import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
import re
import matplotlib as mpl
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colors import Colormap
from svgutils.transform import fromfile, SVGFigure

from style_tools import set_msk_style
set_msk_style()


def save_shap_bar(cfg, sh_mat, feat_names, tag, max_display=20, width=6.5, row_height=0.28):
    # Height proportional to number of rows displayed
    n = min(max_display, len(feat_names))
    height = max(3.0, row_height * n)
    plt.figure(figsize=(width, height), dpi=300)
    shap.summary_plot(sh_mat, pd.DataFrame(sh_mat, columns=feat_names),
                      plot_type="bar", show=False, max_display=max_display)
    plt.title(f"SHAP bar – {tag} (≤{cfg.label_window} m)")
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=8)
    plt.tight_layout()  # Adjust layout to prevent cutoff
    for ext in ("svg", "png"):
        fn = cfg.plots / f"{cfg.outcome}_SHAP_bar_{tag}.{ext}"
        plt.savefig(fn, dpi=300 if ext == "png" else None)
    plt.close()

def save_shap_beeswarm(cfg, sh_mat, X_for_color, tag,
                       max_display=20, row_height=0.28, width=6.5):
    n = min(max_display, X_for_color.shape[1])
    height = max(3.0, row_height * n)
    plt.figure(figsize=(width, height), dpi=300)
    shap.summary_plot(sh_mat, X_for_color, plot_type="dot",
                      show=False, max_display=max_display)
    for ext in ("svg", "png"):
        fn = cfg.plots / f"{cfg.outcome}_SHAP_beeswarm_{tag}.{ext}"
        plt.savefig(fn, dpi=300 if ext == "png" else None, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------ #

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def _shap_cmap(reverse: bool = False):
    """Return the SHAP beeswarm colormap (blue→red), with fallback."""
    try:
        # SHAP ≥0.40
        from shap.plots import colors as shap_colors
        cmap = getattr(shap_colors, "blue_red", None) or getattr(shap_colors, "red_blue", None)
        if cmap is None:  # very old SHAP
            raise ImportError
    except Exception:
        # reasonable fallback if SHAP API changes
        cmap = plt.get_cmap("coolwarm")
    return cmap.reversed() if reverse else cmap

def save_dependence_grid(
    cfg,
    sh_mat: np.ndarray,
    X_for_color: pd.DataFrame,
    feat_names: Sequence[str],
    *,
    base: str,
    tag: str,
    months: Iterable[int] = (0, 12, 24, 36),
    width_per_ax: float = 3,     # inches per panel
    height: float = 4.2,           # inches per row
    dot_size: float = 6,          # point size for scatter
    alpha: float = 0.8,
    cmap: Colormap | None = None,
    exts: Sequence[str] = ("svg", "png"),
) -> Path | None:
    """
    One row of SHAP dependence plots for 'base' across months.
    - Shared x/y limits for comparability
    - One colorbar per row (same vmin/vmax across months)
    - Vector SVG + PNG output
    """
    from style_tools import set_msk_style
    set_msk_style()  # Apply consistent style settings

    if cmap is None:
        cmap = _shap_cmap()  

    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    feats = [f"{base}_m{m}" for m in months
             if f"{base}_m{m}" in X_for_color.columns and f"{base}_m{m}" in name_to_idx]
    if not feats:
        print(f"∙ No features found for base='{base}' → grid skipped.")
        return None

    # Shared ranges across the row
    xvals = [X_for_color[f].to_numpy(float) for f in feats]
    yvals = [sh_mat[:, name_to_idx[f]] for f in feats]
    xmin, xmax = np.nanmin([v.min() for v in xvals]), np.nanmax([v.max() for v in xvals])
    ymin, ymax = np.nanmin([v.min() for v in yvals]), np.nanmax([v.max() for v in yvals])

    # Shared color scale across the row
    cmin = np.nanmin([X_for_color[f].to_numpy(float).min() for f in feats])
    cmax = np.nanmax([X_for_color[f].to_numpy(float).max() for f in feats])
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

    n = len(feats)
    fig_w = width_per_ax * n + 0.5  # Reduced extra space for a slimmer colorbar
    fig_h = height
    fig, axes = plt.subplots(
        1, n, figsize=(fig_w, fig_h), dpi=300, sharey=True,
    )
    if n == 1:
        axes = [axes]

    last_sc = None
    for i, (ax, f) in enumerate(zip(axes, feats)):
        shap.dependence_plot(
            f, sh_mat, X_for_color,
            interaction_index=f, show=False, ax=ax
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        m = re.search(r"_m(\d+)$", f).group(1)
        ax.set_title(f"m{m}", pad=3, fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.tick_params(axis='x', labelsize=8)
        if i == 0:
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylabel(f"SHAP value for\n{base.title()}", labelpad=2, fontsize=8)
            y_ticks = np.linspace(ymin, ymax, 5)
            ax.set_yticks(y_ticks)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Feature value", labelpad=2, fontsize=8)
        sc = next((c for c in ax.collections if isinstance(c, mpl.collections.PathCollection)), None)
        if sc is not None:
            sizes = sc.get_sizes()
            sc.set_sizes(np.full_like(sizes, dot_size))
            sc.set_alpha(alpha)
            sc.set_cmap(cmap)
            sc.set_norm(norm)
            sc.set_edgecolors('black')
            # sc.set_linewidths(0.05)
            sc.set_linewidths(0.1)
            last_sc = sc

    # Slimmer colorbar on the right
    if last_sc is not None:
        cax = fig.add_axes([0.93, 0.20, 0.01, 0.65])  
        cb = fig.colorbar(last_sc, cax=cax)
        cb.set_label("Feature value", fontsize=10)
        cb.ax.tick_params(labelsize=8)  

    # Centered suptitle
    fig.suptitle(base.title(), x=0.5, ha="center", y=0.98, fontsize=12)

    fig.tight_layout(rect=[0.0, 0.01, 0.94, 0.96])
    out_base = cfg.plots / f"{cfg.outcome}_{tag}_SHAP_depend_grid_{_safe_name(base)}"
    for ext in exts:
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches='tight')
    plt.close(fig)
    return out_base.with_suffix(".svg")

# --------------------------------------------------------------- #

def stitch_supp_figure(cfg, out_name="Supp_SHAP_TKR_RF.svg"):
    figs = [
        cfg.plots / f"{cfg.outcome}_SHAP_bar_RF.svg",
        cfg.plots / f"{cfg.outcome}_SHAP_beeswarm_RF.svg",
        cfg.plots / f"{cfg.outcome}_RF_SHAP_depend_grid_lateral_tibial_cartilage.svg",
        cfg.plots / f"{cfg.outcome}_RF_SHAP_depend_grid_lateral_meniscus.svg",
    ]
    svgs = [fromfile(str(p)) for p in figs]
    # simple vertical stack
    widths = [svg.width for svg in svgs]
    W = max(float(w.replace("pt", "").replace("px", "")) for w in widths)  # Convert to float
    Hs = [int(float(s.height.replace("pt", "").replace("px", ""))) for s in svgs]  # Convert height to int
    H = sum(Hs) + 30

    fig = SVGFigure(f"{int(W)}px", f"{H}px")  
    y = 0
    elems = []
    for s in svgs:
        root = s.getroot()
        root.moveto(0, y)
        elems.append(root)
        y += int(float(s.height.replace("pt", "").replace("px", "")))
    fig.append(elems)
    out = cfg.plots / out_name
    fig.save(str(out))
    print("stitched figure ->", out)


def compose_tkr_rf_supp(cfg,
                        bar_path: Path,
                        bees_path: Path,
                        grid1_path: Path,
                        grid2_path: Path,
                        outname=None):
    imgs = [plt.imread(p) for p in [bar_path, bees_path, grid1_path, grid2_path]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    for ax, im, tag in zip(axes.flat, imgs, list("ABCD")):
        ax.imshow(im)
        ax.axis("off")
        ax.text(0.01, 0.99, tag, transform=ax.transAxes,
                fontsize=16, weight="bold", va="top", ha="left")
    fig.tight_layout()
    if outname is None:
        outname = cfg.plots / f"{cfg.outcome}_RF_SHAP_supplement.png"
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Supplement page ->", outname)


