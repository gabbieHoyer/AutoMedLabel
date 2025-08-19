"""
Landmark-based TKR risk-modelling pipeline
=========================================

This one module replaces the scattered notebook cells.  It is fully
parameter-driven: change the numbers in the 'Config' object - or pass a
second 'Config' - and you immediately get a fresh 48m->96m or
48m->120m experiment, including tables + plots written to neat
sub-folders.

Typical usage (inside a notebook or as plain Python):

```python
from landmark_pipeline import Config, run_experiment

might actually be
cfg_96  = Config(label_window=48, pred_window=48, censor_time=120)   # 48m data ->  96-m horizon
# cfg_96  = Config(label_window=48, pred_window=96)   # 48m data ->  96-m horizon


cfg_120 = Config(label_window=48, pred_window=72, censor_time=144)  # 48m data -> 120-m horizon
# cfg_120 = Config(label_window=48, pred_window=120)  # 48m data -> 120-m horizon

run_experiment(cfg_96)
run_experiment(cfg_120)
```

Everything the old notebook did - landmark CV, ROC/AUC CI, decision
curves (standard + custom FP penalties), calibration summaries, final
full-fit models, SHAP summaries/heat-maps, demographics - is wrapped in
clean, testable functions.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
from typing import Iterable, Mapping, Tuple, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from numpy.random import default_rng
import scipy

# from data_utils import (
#     BIOMETRIC_COLS, DEMOGRAPHIC_COLS, TIMEPOINTS_ALL,
#     clean_oai, get_feature_fn, get_event_time_col
# )
# from model_utils import get_lr, get_rf
# from outcomes import get_label_fn

from clinical_utility.landmark_survival import (
    make_feature_matrix_demo_only,
    get_dc_scenarios,
    get_feature_fn, get_event_time_col,
    BIOMETRIC_COLS, DEMOGRAPHIC_COLS, TIMEPOINTS_ALL,
    clean_oai,
    get_lr, get_rf,
    get_label_fn,
    save_shap_bar, save_shap_beeswarm, save_dependence_grid,
    set_msk_style, roc_panel, get_colour,
)

# -*- coding: utf-8 -*-
# --------------------------------------------------------------------
# 1. CONFIGURATION OBJECT
# --------------------------------------------------------------------
@dataclass
class Config:
    """All user-tweakable knobs live here."""

    label_window: int = 48                # max month of features  (LM <= this)
    pred_window: int = 48                 # horizon after LM to call an event
    censor_time: Optional[int] = None     # int = 120            # censor at 12 yr
    landmarks: Tuple[int, ...] = (0, 12, 24, 36, 48)

    biometric_cols: Tuple[str, ...] = BIOMETRIC_COLS
    timepoints:     Tuple[int, ...] = TIMEPOINTS_ALL
    demo_cols: Tuple[str, ...]      = DEMOGRAPHIC_COLS

    outcome: str = "tkr"  # "oa"
    impute: bool = False
    baseline: bool = False  # run demographics only
    save_cv_splits: bool = False

    cv_folds: int = 5
    cv_seed:  int = 42

    # file/plot output roots
    root: Path = Path("./landmark_runs")
    tag: str = field(init=False) 

    def __post_init__(self):
        if self.censor_time is None:
            self.censor_time = self.label_window + self.pred_window

        self.tag = f"{self.outcome}_LM{self.label_window}_H{self.pred_window}_Censor{self.censor_time}"
        if self.baseline:
            self.tag += "_baseline"
        if self.impute:
            self.tag += "_impute"   

        (self.root / self.tag / "tables").mkdir(parents=True, exist_ok=True)
        (self.root / self.tag / "plots").mkdir(parents=True, exist_ok=True)
        (self.root / self.tag / "models").mkdir(parents=True, exist_ok=True)

    # helpers -------------------------------
    @property
    def tables(self) -> Path:  
        return self.root / self.tag / "tables"

    @property
    def plots(self) -> Path:
        return self.root / self.tag / "plots"

    @property
    def models(self) -> Path:
        return self.root / self.tag / "models"

    @property
    def horizon_label(self) -> str:
        return f"{self.pred_window} m"

# --------------------------------------------------------------------
# 3. MODELS & HELPERS
# --------------------------------------------------------------------

def get_models() -> Dict[str, BaseEstimator]:
    return {"RF": get_rf(), "LR": get_lr()}

def safe_predict_proba(clf: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    prob = clf.predict_proba(X)
    return prob[:, 1] if prob.shape[1] == 2 else np.full(len(X), clf.classes_[0])

# --------------------------------------------------------------------
# 4. LANDMARK CROSS-VALIDATION
# --------------------------------------------------------------------

def landmark_loop(df: pd.DataFrame, cfg: Config) -> Tuple[Dict, Dict]:
    stored_oof: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    auc_series: Dict[str, List[float]] = {m: [] for m in get_models()}

    cv = StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.cv_seed)

    for LM in cfg.landmarks:
        visits = [tp for tp in cfg.timepoints if tp <= LM]

        # --- build X_land -------------------------------------------
        if cfg.baseline:
            # from data_utils import make_feature_matrix_demo_only
            X_land = make_feature_matrix_demo_only(
                df, visits, outcome=cfg.outcome, impute=cfg.impute
            )
        else:
            feat_fn = get_feature_fn(cfg.outcome)
            if cfg.outcome == "oa" and cfg.impute:
                feat_fn = get_feature_fn("tkr")
            X_land = feat_fn(df, visits)

        # ------- at-risk filter -------------------------------------
        evt_col = get_event_time_col(cfg.outcome)
        subj_tbl = (df.drop_duplicates("subject_id")
                      .set_index("subject_id")
                      [[evt_col]]
                      .loc[X_land.index])
        keep  = subj_tbl[evt_col] > LM
        X_use = X_land.loc[keep]

        label_fn = get_label_fn(cfg.outcome)
        y_use = label_fn(df, X_use.index, LM, cfg.pred_window)

        # --- track CV design and per-fold RF AUCs for both modes ---
        fold_assign = pd.Series(index=np.arange(len(y_use)), dtype="Int64")
        rf_fold_rows: List[Tuple[int, float]] = []

        for tag, base_clf in get_models().items():
            oof = np.zeros(len(y_use))

            for fold_idx, (tr, te) in enumerate(cv.split(X_use, y_use, groups=X_use.index)):
                clf = clone(base_clf)

                if len(np.unique(y_use.iloc[tr])) > 1:
                    clf.fit(X_use.iloc[tr], y_use.iloc[tr])
                    te_probs = safe_predict_proba(clf, X_use.iloc[te])
                else:
                    te_probs = np.repeat(y_use.iloc[tr].iloc[0], len(te))

                oof[te] = te_probs

                # Record RF as the reference split for saving
                if tag == "RF":
                    fold_assign.iloc[te] = fold_idx
                    if y_use.nunique() == 2:
                        rf_fold_rows.append((fold_idx, float(roc_auc_score(y_use.iloc[te], te_probs))))

            oof_raw = oof.copy()

            # Isotonic calibration
            iso = IsotonicRegression(out_of_bounds="clip").fit(oof_raw, y_use)
            p_cal = iso.transform(oof_raw)

            # keep both
            stored_oof[(tag, LM)] = {
                "proba": p_cal,         
                "proba_raw": oof_raw,     # uncalibrated
                "y_true": y_use.reset_index(drop=True),
            }

            if y_use.nunique() == 2:
                auc_series[tag].append(roc_auc_score(y_use, p_cal))
            else:
                print(f"[LM {LM}] {tag}: single-class ({y_use.iloc[0]}) - AUC skipped")
                auc_series[tag].append(np.nan)

        # --- save design + per-fold AUCs ---
        if len(rf_fold_rows) == cfg.cv_folds:
            kind = "demo" if cfg.baseline else "full"

            # fold,auc per-fold file
            out_folds = cfg.tables / f"{cfg.tag}_RF_{kind}_folds_LM{LM}.csv"
            pd.DataFrame(rf_fold_rows, columns=["fold", "auc"]).to_csv(out_folds, index=False)
            print(f"✓  RF per-fold AUCs -> {out_folds}")

            # fold assignment and X matrix
            if cfg.baseline or cfg.save_cv_splits:
                out_assign = cfg.tables / f"{cfg.tag}_RF_{kind}_fold_assign_LM{LM}.csv"
                pd.DataFrame({
                    "subject_id": X_use.index.to_numpy(),
                    "fold": fold_assign.to_numpy()
                }).to_csv(out_assign, index=False)
                print(f"✓  RF fold assignment -> {out_assign}")

                out_X = cfg.tables / f"{cfg.tag}_X_{kind}_LM{LM}.csv"
                X_use.reset_index().to_csv(out_X, index=False)
                print(f"✓  X_{kind} -> {out_X}")

    return stored_oof, auc_series



# --------------------------------------------------------------------
# 5. COMPLETE ANALYSIS HELPERS  (all side-effects: write tables/plots)
# --------------------------------------------------------------------
# 5.0  save OOF predictions ------------------------------------------
# --------------------------------------------------------------------

def save_oof(cfg: Config, stored_oof: dict):
    """
    Write one CSV per model with all landmarks stacked:

        model,landmark,subject_id,y_true,proba
        RF,0,ID_001,0,0.12
        …
    """
    for tag in get_models():
        rows = []
        for LM in cfg.landmarks:
            d = stored_oof[(tag, LM)]
            rows.append(pd.DataFrame(dict(
                model=tag,
                landmark=LM,
                subject_id=np.arange(len(d["y_true"])),
                y_true=d["y_true"],
                proba=d["proba"],                 # calibrated 
                proba_raw=d.get("proba_raw", np.nan),  
            )))

        out = cfg.tables / f"{cfg.tag}_{tag}_OOF.csv"
        pd.concat(rows).to_csv(out, index=False)
        print("✓  OOF predictions ->", out)

# --------------------------------------------------------------------
# 5.1  calibration summary  (robust: handles single-class + 0/1 probs)
# --------------------------------------------------------------------
def write_calibration_summary(cfg: Config,
                              lm: int,
                              tag: str,
                              y: pd.Series | np.ndarray,
                              p_raw: np.ndarray,
                              p_cal: np.ndarray,
                              eps: float = 1e-6):
    """
    - Writes/append one-row CSV per (landmark, model)
    - Safe when y has only one class or p_cal contains 0/1 values.
    """
    y = np.asarray(y)

    # ---------- guard: single-class ---------------------------------
    if y.ndim == 1 and np.unique(y).size == 1:
        slope     = np.nan
        brier_raw = np.nan
        brier_cal = np.nan
    else:
        # clip to avoid +/- inf logits
        eps = 1e-6
        logit = np.log(np.clip(p_cal, eps, 1-eps) / (1-np.clip(p_cal, eps, 1-eps)))
        try:
            slope = (
                LogisticRegression(fit_intercept=False)
                .fit(logit.reshape(-1, 1), y)      # make X 2D
                .coef_[0][0]
            )
        except ValueError:         # if still unstable -> NaN
            slope = np.nan

        brier_raw = brier_score_loss(y, p_raw)
        brier_cal = brier_score_loss(y, p_cal)

        # calibrated Brier should not be meaningfully worse than raw
        tol = 1e-3  # ignore float noise - round to 3 decimals
        if not np.isnan(brier_raw) and not np.isnan(brier_cal):
            assert brier_cal <= brier_raw + tol, (
                f"Brier(cal) {brier_cal:.4f} > Brier(raw) {brier_raw:.4f} "
                f"at LM {lm}, model {tag}. Check isotonic calibration and OOF setup."
            )

    row = pd.DataFrame(
        dict(
            Landmark_m=[lm],
            Model=[tag],
            calibration_slope=[slope if np.isnan(slope) else round(slope, 3)],
            brier_raw=[None if np.isnan(brier_raw) else round(brier_raw, 3)],
            brier_cal=[None if np.isnan(brier_cal) else round(brier_cal, 3)],
        )
    )
    out = cfg.tables / f"{cfg.outcome}_LM{lm}_calibration_summary.csv"
    row.to_csv(out, mode="a", header=not out.exists(), index=False)


# 5.2  bootstrap AUC CIs (simple percentile, 1000 resamples) -------
def bootstrap_auc_ci(y, p, n_boot=1000, alpha=0.05):
    if y.nunique() < 2:
        return (np.nan, np.nan, np.nan)
    aucs = [roc_auc_score(*resample(y, p)) for _ in range(n_boot)]
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(np.mean(aucs), 3), round(lo, 3), round(hi, 3)

def write_auc_ci_table(cfg: Config, all_auc_rows: List[dict]):
    df = pd.DataFrame(all_auc_rows)
    out = cfg.tables / f"{cfg.tag}_AUC_bootstrapCI_all.csv"
    df.to_csv(out, index=False)
    print("AUC CI table ->", out)


# 5.3  ROC plot for every landmark / model -------------------------
def plot_landmark_rocs(cfg: Config,
                       stored_oof: dict,
                       auc_lookup: dict  # (Model,LM) -> (AUC,lo,hi)
                       ):
    fpr_base = np.linspace(0, 1, 100)
    palette  = sns.color_palette("viridis", n_colors=len(cfg.landmarks))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    for ax, tag in zip(axes, ("RF", "LR")):
        for idx, LM in enumerate(cfg.landmarks):
            d = stored_oof[(tag, LM)]
            if d["y_true"].nunique() < 2:
                continue

            # -------- grab stats that were already bootstrapped
            auc, lo, hi = auc_lookup[(tag, LM)]

            fpr, tpr, _ = roc_curve(d["y_true"], d["proba"])
            ax.plot(fpr_base,
                    np.interp(fpr_base, fpr, tpr),
                    color=palette[idx],
                    lw=1.8,
                    label=(
                        f"LM {LM} m  "
                        f"(AUC {auc:.2f} [{lo:.2f}-{hi:.2f}])"
                    ))

        ax.plot([0, 1], [0, 1], ls=":", c="grey")
        ax.set_title(f"{tag} ROC")
        ax.set_xlabel("False-Positive Rate")
        ax.set_ylabel("True-Positive Rate")
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    for ext in ("png", "svg"): #"pdf"):
        fn = cfg.plots / f"{cfg.outcome}_landmark_ROC_panels.{ext}"
        fig.savefig(fn, dpi=300 if ext == "png" else None)

    plt.close(fig)

# ------------------------------------------------------------------
# 5.4  Decision-curve helpers     (outcome-agnostic)
# ------------------------------------------------------------------
def _net_benefit(probs: np.ndarray,
                 events: np.ndarray,
                 thr: np.ndarray,
                 fp_weight: float = None) -> np.ndarray:
    """
    Net-benefit per threshold.
    - If fp_weight is None -> standard formulation.
    - Else subtract custom weight x FP.
    """
    n   = len(events)
    tp  = ((probs >= thr[:, None]) &  events).sum(1) / n
    fp  = ((probs >= thr[:, None]) & ~events).sum(1) / n
    nb  = tp - fp * thr / (1 - thr)
    if fp_weight:
        nb -= fp_weight * fp          # custom penalty
    return nb

# --------- Bootstrap limits for decision curves -------------------
def _nb_ci(y: np.ndarray,
           p: np.ndarray,
           thr: np.ndarray,
           w: float,
           n_boot: int = 5000, #2000,
           bca: bool   = False,
           seed: int   = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Percentile (or BCa) 95% limits for NB(t) at all thresholds.
    Returns (low, high) arrays with shape = len(thr).
    """
    rng   = default_rng(seed)
    n     = y.size
    boot  = np.empty((n_boot, thr.size))

    # pre-compute for the original sample (needed if BCa)
    nb_orig = _net_benefit(p, y, thr, fp_weight=w)

    for b in range(n_boot):
        idx   = rng.choice(n, size=n, replace=True)
        nb_b  = _net_benefit(p[idx], y[idx], thr, fp_weight=w)
        boot[b] = nb_b

    low  = np.percentile(boot, 2.5, axis=0)
    high = np.percentile(boot, 97.5, axis=0)

    if not bca:
        return low, high

    # ----------------- optional BCa adjustment --------------------
    # bias-correction z0
    z0   = scipy.stats.norm.ppf((boot < nb_orig).mean(axis=0))
    # jackknife acceleration (a)   
    jk   = np.array([_net_benefit(np.delete(p, i),
                                  np.delete(y, i),
                                  thr, fp_weight=w)
                     for i in range(n)])
    jk_bar = jk.mean(axis=0)
    a      = ( (jk_bar - jk)**3 ).sum(axis=0) / \
             ( 6.0 * ((jk_bar - jk)**2).sum(axis=0) )
    # adjusted percentiles
    z_low, z_high = scipy.stats.norm.ppf([0.025, 0.975])
    pct_low  = scipy.stats.norm.cdf(z0 + (z0 + z_low )/(1 - a*(z0+z_low ))) * 100
    pct_high = scipy.stats.norm.cdf(z0 + (z0 + z_high)/(1 - a*(z0+z_high))) * 100
    low_bca  = np.percentile(boot, pct_low , axis=0)
    high_bca = np.percentile(boot, pct_high, axis=0)
    return low_bca, high_bca

def _save_dc_csv(cfg: Config,
                 tag: str,
                 scen: str,
                 thr: np.ndarray,
                 nb: np.ndarray,
                 nb_low: np.ndarray,
                 nb_high: np.ndarray) -> None:
    out = pd.DataFrame({
        "threshold":   thr,
        "net_benefit": nb,
        "nb_low":      nb_low,
        "nb_high":     nb_high,
    })
    out.to_csv(cfg.tables / f"{cfg.tag}_{tag}_{scen}.csv",
               index=False)


def _plot_dc_block(cfg: Config,
                   stored_oof: dict,
                   scen_dict: dict[str, float],
                   suffix: str,
                   title_extra: str,
                   ls_map: dict[str, str]) -> None:
    """One figure containing RF & LR for the given scenario set."""
    thr   = np.linspace(0.01, 0.50, 50)
    color = {"RF": "#1f77b4", "LR": "#d62728"}

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)

    for tag in ("RF", "LR"):
        y = stored_oof[(tag, cfg.label_window)]["y_true"]
        p = stored_oof[(tag, cfg.label_window)]["proba"]

        if y.nunique() < 2:
            print(f"∙ decision curves skipped for {tag} - only one class")
            continue

        for scen, fpw in scen_dict.items():
            nb = _net_benefit(p, y.values, thr, fp_weight=fpw)

            # ---------- confidence limits  ----------
            nb_lo, nb_hi = _nb_ci(y.values, p, thr, w=fpw,
                                n_boot=2000, seed=42)
            _save_dc_csv(cfg, tag, scen, thr, nb, nb_lo, nb_hi)

            ax.plot(thr, nb,
                    color=color[tag], ls=ls_map[scen], lw=1.8,
                    label=f"{tag} {scen.replace('_', ' ')}")

    for w, ls, txt in [(1.0, ':',  'Treat all (w = 1)'),
                    (0.20, '--', 'Treat all (w = 0.20)')]:
        ta = y.mean() - w * (1 - y.mean()) * thr / (1 - thr)
        ax.plot(thr, ta, ls=ls, lw=1.0, color='grey', label=txt)

    ax.axhline(0, ls=':', c='k', lw=0.8, label='Treat-none')

    ax.set_xlabel(f"Threshold probability for {cfg.outcome.upper()} "
                  f"≤ {cfg.pred_window} m")
    ax.set_ylabel("Net benefit per patient")
    ax.set_title(f"Decision-curves {title_extra} "
                 f"({cfg.label_window} m window)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    for ext in ("png", "svg"): #"pdf"):
        fn = cfg.plots / f"{cfg.tag}_DC_{suffix}.{ext}"
        fig.savefig(fn, dpi=300 if ext == "png" else None)
 
    plt.close(fig)

    print("∙ decision-curve plot ->", fn)

def write_decision_curves(cfg: Config,
                          stored_oof: dict) -> None:
    """Create two panels: (i) standard, (ii) custom FP weights."""
    scen_all = get_dc_scenarios(cfg.outcome)   # <- pulled from data_utils

    # panel 1 - only the standard curve
    std = {"Standard_FP": scen_all["Standard_FP"]}
    _plot_dc_block(cfg, stored_oof,
                   scen_dict   = std,
                   suffix      = "standard",
                   title_extra = "· standard FP penalty",
                   ls_map      = {"Standard_FP": "-"})

    # panel 2 - all non-standard scenarios for this outcome
    custom = {k: v for k, v in scen_all.items() if k != "Standard_FP"}
    ls_custom = {k: "-" if i == 0 else "--"
                 for i, k in enumerate(custom)}  # simple linestyle map
    _plot_dc_block(cfg, stored_oof,
                   scen_dict   = custom,
                   suffix      = "custom",
                   title_extra = "· custom FP penalties",
                   ls_map      = ls_custom)

# ------------------------------------------------------------------
# ----- "Usable threshold" clinical ranges   (Δ-margin aware) ------
# ------------------------------------------------------------------
# from data_utils import get_dc_scenarios   

def _clinical_ranges(cfg: Config,
                     stored_oof: dict,
                     delta: float = 0.002) -> pd.DataFrame:
    """
    For each (Model, Scenario) report threshold band where  
        NB_model  >  Treat-all + Δ   and  NB_model > Δ.  
    Saves CSV and echoes quick summary to console.
    """
    thr       = np.linspace(0.01, 0.50, 50)
    scen_def  = get_dc_scenarios(cfg.outcome)        # <- generic now
    rows      = []

    for tag in ("RF", "LR"):
        y = stored_oof[(tag, cfg.label_window)]["y_true"]
        p = stored_oof[(tag, cfg.label_window)]["proba"]

        if y.nunique() < 2:
            continue

        n         = len(y)
        prev      = y.mean()
        treat_all = prev - (1 - prev) * thr / (1 - thr)

        # pre-compute rates for speed
        tp_rate = np.asarray([((p >= t) &  y).sum() / n for t in thr])
        fp_rate = np.asarray([((p >= t) & ~y).sum() / n for t in thr])

        for scen, w in scen_def.items():
            if w == 0.0:                       # standard formula
                nb = tp_rate - fp_rate * thr / (1 - thr)
            else:                              # custom FP penalty
                nb = tp_rate - w * fp_rate

            ok  = (nb > delta) & (nb > treat_all + delta)
            rng = f"{thr[ok][0]:.0%} - {thr[ok][-1]:.0%}" if ok.any() else "-"
            rows.append(dict(Model=tag, Scenario=scen, Range=rng))

    out_df = pd.DataFrame(rows)

    # -------- console echo -------------------------------------
    def _pick(model, scen):
        r = out_df.query("Model==@model & Scenario==@scen").Range
        return r.iat[0] if len(r) else "-"

    for scen in scen_def:       
        print(f"LR {scen:<12}: {_pick('LR', scen)}")
        print(f"RF {scen:<12}: {_pick('RF', scen)}")

    fn = cfg.tables / f"{cfg.tag}_usable_threshold_ranges.csv"
    out_df.to_csv(fn, index=False)
    print("Clinical threshold ranges ->", fn)
    return out_df

# ------------------------------------------------------------------
# 5.5  SHAP summaries  (RF & LR) - bar + month-pivot heat-map
# ------------------------------------------------------------------
# from shap_plots import save_shap_bar, save_shap_beeswarm, save_dependence_grid  # stitch_supp_figure removed

def run_shap(cfg: Config, df: pd.DataFrame) -> None:
    # 1) feature matrix up to landmark window
    tp_use = [tp for tp in cfg.timepoints if tp <= cfg.label_window]

    if cfg.baseline:
        # from data_utils import make_feature_matrix_demo_only
        X_full = make_feature_matrix_demo_only(df, tp_use, outcome=cfg.outcome, impute=cfg.impute)
    else:
        feat_fn = get_feature_fn(cfg.outcome)
        if cfg.outcome == "oa" and cfg.impute:
            feat_fn = get_feature_fn("tkr")
        X_full = feat_fn(df, tp_use)

    # 2) keep subjects at risk
    evt_col = get_event_time_col(cfg.outcome)
    subj_tbl = (df.drop_duplicates("subject_id").set_index("subject_id")[[evt_col]].loc[X_full.index])
    keep = subj_tbl[evt_col] > cfg.label_window
    X_full = X_full.loc[keep]

    # 3) labels
    label_fn = get_label_fn(cfg.outcome)
    y_full = label_fn(df, at_risk_idx=X_full.index,
                      landmark_month=cfg.label_window, horizon_months=cfg.pred_window)
    if y_full.nunique() < 2:
        print("∙ SHAP skipped - only one class in y_full")
        return

    # 4) models
    base_models = get_models()
    models = {k: clone(v) for k, v in base_models.items() if k in ("RF", "LR")}

    for tag, mdl in models.items():
        mdl.fit(X_full, y_full)

        if tag == "RF":
            imp = mdl.named_steps["simpleimputer"]
            rf = mdl.named_steps["randomforestclassifier"]
            X_rf = pd.DataFrame(imp.transform(X_full), index=X_full.index, columns=X_full.columns)
            expl = shap.TreeExplainer(rf)
            sh_raw = expl.shap_values(X_rf, check_additivity=False)
        else:
            imp = mdl.named_steps["simpleimputer"]
            scaler = mdl.named_steps["standardscaler"]
            lr_clf = mdl.named_steps["logisticregression"]
            X_std1 = pd.DataFrame(imp.transform(X_full), index=X_full.index, columns=X_full.columns)
            X_std = pd.DataFrame(scaler.transform(X_std1), index=X_std1.index, columns=X_std1.columns)
            expl = shap.LinearExplainer(lr_clf, X_std, feature_perturbation="interventional")
            sh_raw = expl(X_std).values

        # class-1 SHAP values
        if isinstance(sh_raw, list):
            sh_mat = sh_raw[1]
        elif isinstance(sh_raw, np.ndarray) and sh_raw.ndim == 3:
            sh_mat = sh_raw[:, :, 1]
        else:
            sh_mat = sh_raw

        feat_names = list(X_full.columns)
        sh_df = pd.DataFrame(sh_mat, columns=feat_names)

        # pick matrices used for correlation and coloring
        if tag == "RF":
            X_dir = X_rf
            X_for_color = X_rf
        else:
            X_dir = X_std
            X_for_color = X_std1

        sh_df.index = X_dir.index

        # Use helper functions for beeswarm and bar
        save_shap_beeswarm(cfg, sh_mat, X_for_color, tag=tag, max_display=22)
        save_shap_bar(cfg, sh_mat, feat_names, tag=tag)

        # Direction table
        sh_df = sh_df.apply(pd.to_numeric, errors="coerce")
        X_dir = X_dir.apply(pd.to_numeric, errors="coerce")
        nz = X_dir.nunique(dropna=False) > 1
        X_dir_nz = X_dir.loc[:, nz]
        sh_df_nz = sh_df.loc[:, nz]
        corr_spear = sh_df_nz.corrwith(X_dir_nz, method="spearman")
        corr_pear = sh_df_nz.corrwith(X_dir_nz, method="pearson")
        abs_raw = np.abs(sh_mat).mean(axis=0)
        abs_raw_s = pd.Series(abs_raw, index=feat_names)

        dir_tbl = (pd.DataFrame({
                    "feature": list(X_dir.columns),
                    "mean_abs_SHAP_raw": abs_raw_s.reindex(X_dir.columns).values,
                    "spearman_corr": corr_spear.reindex(X_dir.columns),
                    "pearson_corr": corr_pear.reindex(X_dir.columns),
                })
                .assign(direction_hint=lambda d: np.where(
                    d["spearman_corr"] > 0, "higher value -> higher risk",
                    np.where(d["spearman_corr"] < 0, "lower value -> higher risk", "weak/no monotone link")
                ))
                .sort_values("mean_abs_SHAP_raw", ascending=False))
        dir_csv = cfg.tables / f"{cfg.outcome}_SHAP_direction_{tag}.csv"
        dir_tbl.to_csv(dir_csv, index=False)

        # Dependence grids - example features
        if cfg.outcome == "tkr" and tag == "RF":
            save_dependence_grid(cfg, sh_mat, X_for_color, feat_names,
                                 base="lateral tibial cartilage", tag=tag,
                                 months=(0, 12, 24, 36), width_per_ax=3.2, height=2.6, dot_size=18)
            save_dependence_grid(cfg, sh_mat, X_for_color, feat_names,
                                 base="lateral meniscus", tag=tag,
                                 months=(0, 12, 24, 36), width_per_ax=3.2, height=2.6, dot_size=18)

        if cfg.outcome == "oa" and tag == "RF":
            save_dependence_grid(cfg, sh_mat, X_for_color, feat_names,
                                 base="lateral tibial cartilage", tag=tag,
                                 months=(0, 12, 24, 36), width_per_ax=3.2, height=2.6, dot_size=18)
            save_dependence_grid(cfg, sh_mat, X_for_color, feat_names,
                                 base="medial meniscus", tag=tag,
                                 months=(0, 12, 24, 36), width_per_ax=3.2, height=2.6, dot_size=18)

        if cfg.outcome == "oa" and tag == "LR":
            save_dependence_grid(cfg, sh_mat, X_for_color, feat_names,
                                 base="patellar cartilage", tag=tag,
                                 months=(0, 12, 24, 36), width_per_ax=3.2, height=2.6, dot_size=18)
            save_dependence_grid(cfg, sh_mat, X_for_color, feat_names,
                                 base="lateral tibial cartilage", tag=tag,
                                 months=(0, 12, 24, 36), width_per_ax=3.2, height=2.6, dot_size=18)

        # Per-feature importance table
        abs_norm = abs_raw / abs_raw.sum()
        (pd.DataFrame({
            "feature": feat_names,
            "mean_abs_SHAP_raw": abs_raw,
            "mean_abs_SHAP": abs_norm
        })
         .sort_values("mean_abs_SHAP_raw", ascending=False)
         .to_csv(cfg.tables / f"{cfg.outcome}_SHAP_overall_{tag}.csv", index=False))

        # Month-pivot heat map (keep here)
        df_imp = (pd.DataFrame({"feature": feat_names, "abs_raw": abs_raw, "abs_norm": abs_norm})
                  .query("feature.str.contains('_m')", engine="python")
                  .assign(
                      base=lambda d: d.feature.str.replace(r"_m\d+$", "", regex=True),
                      month=lambda d: d.feature.str.extract(r"_m(\d+)$")[0].astype(int)
                  ))
        if df_imp.empty:
            print(f"SHAP heat-map skipped for {tag} - no time-stamped feats")
            continue

        pivot = (df_imp.pivot_table(index="base", columns="month", values="abs_norm",
                                    aggfunc="sum", fill_value=0)
                 .reindex(cfg.biometric_cols, fill_value=0)
                 .reindex(sorted(df_imp.month.unique()), axis=1))
        df_imp.to_csv(cfg.tables / f"{cfg.outcome}_SHAP_long_{tag}.csv", index=False)
        pivot.to_csv(cfg.tables / f"{cfg.outcome}_SHAP_pivot_{tag}.csv")

        sns.heatmap(pivot, cmap="mako", cbar_kws={'label': 'mean |SHAP|'})
        plt.ylabel("Biometric"); plt.xlabel("Month")
        plt.title(f"Feature importance across time - {tag}")
        plt.tight_layout()
        for ext in ("png", "svg"):
            hm_fn = cfg.plots / f"{cfg.outcome}_SHAP_heatmap_{tag}.{ext}"
            plt.savefig(hm_fn, dpi=300 if ext == "png" else None)
        plt.close()



# --------------------------------------------------------------------
# 5.6  landmark demographics  (robust to missing baseline visits)
# --------------------------------------------------------------------
# from data_utils import get_feature_fn, get_event_time_col   

def write_demographics(cfg: Config, df: pd.DataFrame) -> None:
    """
    One row per landmark with:
        - N at-risk knees   (no event before LM)
        - # events within forecast horizon
        - % events
        - mean age / BMI, % female
    Works even when a subject has no month-0 visit.
    """
    # ----------  baseline demo table (first visit per subject)  -----
    base = (df.sort_values("months")
              .drop_duplicates("subject_id")
              .set_index("subject_id")[["age", "sex", "BMI"]])

    evt_col = get_event_time_col(cfg.outcome)              # <─ generic

    rows = []
    for LM in cfg.landmarks:
        visits   = [tp for tp in cfg.timepoints if tp <= LM]

        # ---------  feature matrix for the landmark window  ---------
        if cfg.baseline:
            # align cohort to the outcome-specific builder, then keep only demos
            # from data_utils import make_feature_matrix_demo_only

            at_risk = make_feature_matrix_demo_only(
                df, visits, outcome=cfg.outcome, impute=cfg.impute
            ).index
        else:
            feat_fn = get_feature_fn(cfg.outcome)
            if cfg.outcome == "oa" and cfg.impute:
                feat_fn = get_feature_fn("tkr")
            at_risk  = feat_fn(df, visits).index


        first_evt = (df.drop_duplicates("subject_id")
                       .set_index("subject_id")
                       .loc[at_risk, evt_col])             # generic column

        keep_mask     = first_evt > LM                     # still event-free
        at_risk_kept  = at_risk[keep_mask]

        n      = len(at_risk_kept)
        events = (first_evt[keep_mask] <= LM + cfg.pred_window).sum()

        demo   = base.reindex(at_risk_kept)                # NaNs auto-ignored

        rows.append(dict(
            Landmark_m = LM,
            At_risk_N  = n,
            Events     = int(events),
            Event_pct  = round(100 * events / n, 1) if n else 0.0,
            Age_mean   = round(demo["age"].mean(), 1),
            Female_pct = round(100 * (demo["sex"] == 2).mean(), 1),
            BMI_mean   = round(demo["BMI"].mean(), 1),
        ))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(cfg.tables / f"{cfg.outcome}_landmark_demographics.csv",
                  index=False)
    print("Demographics written ->", cfg.tables)


# ------------------------------------------------------------------
# 5.8  Experiment-summary table (AUC/Cal  | DC ranges | Demographics)
# ------------------------------------------------------------------
def _write_experiment_summary(cfg: Config) -> None:
    """
    Build three blocks (discrimination+calibration · decision-curve ranges
    - landmark demographics) and stack them into one master CSV.

    The decision-curve section is now outcome-agnostic:
       - reads scenarios from cfg.dc_scenarios  (e.g. {"Standard_FP":0,
                                                      "MRI_FP0.2":0.2, …})
       - prettifies names with cfg.dc_labels    (optional helper dict)
    """
    from tabulate import tabulate

    # ---------- 1. DISCRIMINATION & CALIBRATION -------------------
    auc_ci = pd.read_csv(
        cfg.tables / f"{cfg.tag}_AUC_bootstrapCI_all.csv"
    )
    discr_rows = []
    for lm in cfg.landmarks:
        perf = pd.read_csv(
            cfg.tables / f"{cfg.outcome}_LM{lm}_calibration_summary.csv"
        )
        for tag, long_name in [("RF", "Random forest"),
                               ("LR", "Logistic regression")]:
            slope, br_raw, br_cal = perf.query("Model==@tag").iloc[-1][
                ["calibration_slope", "brier_raw", "brier_cal"]
            ]
            auc_row = auc_ci.query("Model==@tag & Landmark_m==@lm").iloc[0]
            discr_rows.append(
                dict(Stage="Discrimination & calibration",
                     Landmark_m=lm,
                     Model=long_name,
                     AUC=round(auc_row.AUC, 3),
                     CI_low=round(auc_row.CI_low, 3),
                     CI_high=round(auc_row.CI_high, 3),
                     CalSlope=None if np.isnan(slope) else round(slope, 3),
                     Brier_raw=br_raw,
                     Brier_cal=br_cal)
            )
    discrim_tbl = pd.DataFrame(discr_rows)
    discrim_tbl.to_csv(
        cfg.tables / f"{cfg.tag}_summary_discrim.csv",
        index=False
    )

    # ---------- 2. DECISION-CURVE RANGES (agnostic) ---------------
    range_tbl = pd.read_csv(
        cfg.tables / f"{cfg.tag}_usable_threshold_ranges.csv"
    )
    # --------------------------------------- decision-curve section
    scen_penalties = get_dc_scenarios(cfg.outcome)          # raw -> weight

    # 1) prettify 
    pretty_map = {
        "Standard_FP": "Standard FP penalty",
        "MRI_FP0.2"  : "MRI triage (FP 0.20)",
        "Surg_FP1.0" : "Surgery triage (FP 1.00)",
        "PrevCounsel_FP1.0": "Preventive counselling (FP 1.00)",
    }
    range_tbl["Scenario"] = range_tbl["Scenario"].map(
        lambda k: pretty_map.get(k, k.replace("_", " "))
    )

    # 2) derive a matching dict: pretty label -> weight
    pretty_to_weight = {pretty_map.get(k, k.replace("_", " ")): w
                        for k, w in scen_penalties.items()}

    range_piv = (range_tbl
                .pivot(index="Scenario", columns="Model", values="Range")
                .rename(columns={"RF": "Net-benefit RF",
                                "LR": "Net-benefit LR"})
                .reset_index())

    # 3) look-up succeeds
    range_piv["Treat-all"] = [
        "baseline" if pretty_to_weight[s] == 0 else "negative baseline"
        for s in range_piv["Scenario"]
    ]
    range_piv["Treat-none"] = ["0"] * len(range_piv)


    range_piv.to_csv(
        cfg.tables / f"{cfg.tag}_summary_dcurves.csv",
        index=False
    )

    # ---------- 3. DEMOGRAPHICS  ---------------------------------
    demo_tbl = (pd.read_csv(cfg.tables /
                            f"{cfg.outcome}_landmark_demographics.csv")
                  .query("Landmark_m in @cfg.landmarks")
                  .assign(Stage="Landmark demographics"))
    demo_tbl.to_csv(
        cfg.tables / f"{cfg.tag}_summary_demo.csv",
        index=False
    )

    # ---------- 4. STACK & SAVE  ---------------------------------
    stacked = pd.concat([discrim_tbl, range_piv, demo_tbl], ignore_index=True)
    stacked.to_csv(
        cfg.tables / f"{cfg.tag}_summary_ALL.csv",
        index=False
    )

    # ---------- 5. Pretty-print to console ------------------------
    def show(df, title):
        print(f"\n» {title}")
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

    print("\n===   EXPERIMENT SUMMARY  ==================================")
    show(discrim_tbl,   "Discrimination & calibration")
    show(range_piv,     "Decision-curve usable ranges")
    show(demo_tbl,      "Landmark demographics")
    print("\n Summary tables written ->", cfg.tables, "\n")

# --------------------------------------------------------------------
# 6. MAIN ENTRY POINT - now calls every helper above
# --------------------------------------------------------------------
def run_experiment(cfg: Config, df_raw: pd.DataFrame | None = None):
    if df_raw is None:
        raise ValueError("Pass the cleaned OAI dataframe as df_raw")

    df = clean_oai(df_raw, outcome=cfg.outcome, censor_time=cfg.censor_time)

    # 6.1  landmark CV  ------------------------------------------------
    stored_oof, auc_dict = landmark_loop(df, cfg)
    save_oof(cfg, stored_oof)   

    # 6.2  bootstrap CIs & calibration summaries -----------------------
    auc_rows = []
    for (tag, LM), d in stored_oof.items():
        p_cal = d["proba"]                          # calibrated
        p_raw = d.get("proba_raw", d["proba"])      # fallback if needed
        y     = d["y_true"]

        # AUC uses calibrated probabilities (as before)
        auc, lo, hi = bootstrap_auc_ci(y, p_cal)
        auc_rows.append(dict(Model=tag, Landmark_m=LM, AUC=auc, CI_low=lo, CI_high=hi))

        # Calibration summary: raw vs calibrated Brier split correctly
        write_calibration_summary(cfg, LM, tag, y, p_raw, p_cal)

    write_auc_ci_table(cfg, auc_rows)

    # 6.3  plots & decision curves ------------------------------------
    auc_lookup = {(r["Model"], r["Landmark_m"]): (r["AUC"],
                                                r["CI_low"],
                                                r["CI_high"])
                for r in auc_rows}

    # 6.3  plots & decision curves ------------------------------------
    plot_landmark_rocs(cfg, stored_oof, auc_lookup)   

    write_decision_curves(cfg, stored_oof)

    # 6.4  SHAP summary + demographics --------------------------------
    run_shap(cfg, df)
    write_demographics(cfg, df)

    # 6.5  clinically usable threshold ranges -------------------------
    _clinical_ranges(cfg, stored_oof, delta=0.002)   # Δ=0 (og) or 0.002

    # 6.6  one-pager summary table -----------------------------------
    _write_experiment_summary(cfg)

# --------------------------------------------------------------------
# 7 · If executed as script -------------------------------------------
# --------------------------------------------------------------------
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Landmark TKR / OA inc. risk-model runner")
    p.add_argument("--outcome", default="tkr", choices=["tkr", "oa"])
    p.add_argument("--label_window", type=int, default=48)
    p.add_argument("--pred_window", type=int, default=48)
    p.add_argument("--censor_time", type=int)
    p.add_argument("--oai_pickle", type=str, required=True,
                   help="Pickle/feather/... file with the pre-cleaned OAI dataframe")
    p.add_argument("--impute", action="store_true",
                   help="Use mean-imputation and keep rows with missing biomarker values")
    p.add_argument("--baseline", action="store_true",
                   help="Use demographics-only baseline (age, sex, BMI)")
    p.add_argument("--save_cv_splits", action="store_true",
                   help="Write CV fold assignment CSVs for each landmark")
    p.add_argument("--run_all", action="store_true",
                   help="Run the full set of predefined experiments")
    args = p.parse_args()

    # Load once and reuse
    df = pd.read_pickle(args.oai_pickle)

    def run_once(cfg_kwargs):
        cfg = Config(
            label_window=cfg_kwargs.get("label_window", 48),
            pred_window=cfg_kwargs["pred_window"],
            censor_time=cfg_kwargs["censor_time"],
            outcome=cfg_kwargs["outcome"],
            impute=cfg_kwargs.get("impute", False),
            baseline=cfg_kwargs.get("baseline", False),
            save_cv_splits=args.save_cv_splits,
        )
        print(f"\n=== Running {cfg.tag}  "
              f"(baseline={cfg.baseline}, impute={cfg.impute}) ===")
        run_experiment(cfg, df)

    if args.run_all:
        runs = [
            # 48 -> 96 horizon, demographics-only
            dict(outcome="tkr", pred_window=48, censor_time=120, baseline=True),
            dict(outcome="oa",  pred_window=48, censor_time=120, baseline=True),

            # 48 -> 96 horizon, biomarker models
            dict(outcome="tkr", pred_window=48, censor_time=120),
            dict(outcome="oa",  pred_window=48, censor_time=120),

            # 48 -> 120 horizon, biomarker models
            dict(outcome="tkr", pred_window=72, censor_time=144),
            dict(outcome="oa",  pred_window=72, censor_time=144),

            # OA mean-imputed sensitivity runs
            dict(outcome="oa",  pred_window=48, censor_time=120, impute=True),
            dict(outcome="oa",  pred_window=72, censor_time=144, impute=True),
        ]

        for i, spec in enumerate(runs, 1):
            print(f"\n[{i}/{len(runs)}]")
            run_once(spec)
    else:
        # Single run, honoring CLI flags
        run_once(dict(
            outcome=args.outcome,
            pred_window=args.pred_window,
            censor_time=args.censor_time,
            baseline=args.baseline,
            impute=args.impute,
            label_window=args.label_window,
        ))


# python landmark_pipeline.py \
#     --oai_pickle /data/msk_infocommons/Users/ghoyer/shoulder/PrecisionShoulderAI/knee_inference/OAI_landmark_analysis_0702/oa_filtered_clean.pkl \
#     --run_all


# python -m clinical_utility.run_landmark_pipeline
