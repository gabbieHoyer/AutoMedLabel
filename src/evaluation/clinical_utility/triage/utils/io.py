from pathlib import Path
import pandas as pd

def make_out_dir(cfg) -> Path:
    base = Path(cfg["data"]["out_dir"])
    tag = cfg["tag"]

    # grab thresholds and turn into integer percentages
    th = cfg.get("thresholds", {})
    a = th.get("stage_a_spec")
    b = th.get("stage_b_to_c_spec")
    if a is not None and b is not None:
        tag = f"a{int(a*100)}p_b{int(b*100)}p"

    out = base / tag
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_csv(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg["data"]["csv_path"])

def save_table(df: pd.DataFrame, name: str, cfg):
    out = make_out_dir(cfg) / f"{name}.csv"
    df.to_csv(out, float_format="%.6f")
    return out

def export_svg(fig, fname, target_w_cm=17.8, dpi=300):
    """
    Save an SVG whose width equals 'target_w_cm' and whose height
    keeps the original aspect ratio.
    """
    w_in, h_in = fig.get_size_inches()
    fig.set_size_inches(target_w_cm / 2.54,
                        h_in * target_w_cm / 2.54 / w_in,
                        forward=True)
    fig.savefig(fname, format="svg",
                bbox_inches="tight", pad_inches=0.01, dpi=dpi)

def save_fig(fig, name: str, cfg,
             png=True, svg=False, svg_cm=17.8, **kw):
    """
    Wrapper for both PNG and (optionally) SVG.

    Parameters
    ----------
    fig       : matplotlib Figure
    name      : stem without extension
    cfg       : your experiment cfg (needs data.out_dir + tag)
    png       : write PNG when True
    svg       : write SVG when True
    svg_cm    : target width in cm for the SVG
    **kw      : forwarded to fig.savefig for the PNG
    """
    out_dir = make_out_dir(cfg)

    if png:
        fig.savefig(out_dir / f"{name}.png",
                    dpi=300, bbox_inches="tight", **kw)
    if svg:
        export_svg(fig, out_dir / f"{name}.svg",
                   target_w_cm=svg_cm)
