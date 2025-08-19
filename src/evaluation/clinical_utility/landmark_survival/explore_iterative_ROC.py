#!/usr/bin/env python
"""
Exploratory figure: iteratively add visits (0 -> 96 m) and plot
cross-validated ROC curves for one RF model.
Uses the same Config / cleaning helpers as the main pipeline.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample

from landmark_pipeline import Config
from data_utils import (
    clean_oai, get_feature_fn, BIOMETRIC_COLS, DEMOGRAPHIC_COLS,
)
from outcomes import get_label_fn
from model_utils import get_rf   #  <- canonical RF pipeline

# ------------------------------------------------------------------ #
# 1. helper – one RF, percentile C-I                                 #
# ------------------------------------------------------------------ #
def _bootstrap_auc(y, p, n=1000, alpha=0.05):
    if y.nunique() < 2:
        return (np.nan, np.nan, np.nan)
    aucs = [roc_auc_score(*resample(y, p)) for _ in range(n)]
    lo, hi = np.percentile(aucs, [100*alpha/2, 100*(1-alpha/2)])
    return round(np.mean(aucs), 3), round(lo, 3), round(hi, 3)

# ------------------------------------------------------------------ #
# 2. main                                                            #
# ------------------------------------------------------------------ #
def run_iterative_roc(cfg: Config,
                      df_raw: pd.DataFrame,
                      visit_sets: List[Tuple[int, ...]]) -> None:

    df        = clean_oai(df_raw, outcome=cfg.outcome,
                          censor_time=cfg.censor_time)
    feat_fn   = get_feature_fn(cfg.outcome)
    if cfg.outcome == "oa" and cfg.impute:
        # temporarily reuse the TKR builder
        feat_fn = get_feature_fn("tkr")
    
    label_fn  = get_label_fn(cfg.outcome)

    base_fpr  = np.linspace(0, 1, 100)
    palette   = plt.get_cmap("viridis")(np.linspace(0, 1, len(visit_sets)))

    rows_csv  = []                                     # AUC table rows
    plt.figure(figsize=(8, 6), dpi=110)

    for idx, visits in enumerate(visit_sets):
        # ---- feature matrix & label -----------------------------
        X = feat_fn(df, visits)

        y = label_fn(df,
                     at_risk_idx=X.index,
                     landmark_month=0,
                     horizon_months=cfg.pred_window)

        cols = [f"{c}_m{tp}" for tp in visits for c in BIOMETRIC_COLS] \
             + list(DEMOGRAPHIC_COLS)
        X_use = X[cols]

        cv     = StratifiedGroupKFold(n_splits=cfg.cv_folds,
                                      shuffle=True, random_state=cfg.cv_seed)
        oof    = np.zeros(len(y))
        mean_t = np.zeros_like(base_fpr)

        for tr, te in cv.split(X_use, y, groups=X_use.index):
            rf = get_rf(seed=cfg.cv_seed)              # canonical model
            rf.fit(X_use.iloc[tr], y.iloc[tr])
            prob       = rf.predict_proba(X_use.iloc[te])[:, 1]
            oof[te]    = prob
            fpr, tpr,_ = roc_curve(y.iloc[te], prob)
            mean_t    += np.interp(base_fpr, fpr, tpr)

        mean_t /= cv.get_n_splits()
        auc, lo, hi = _bootstrap_auc(y, oof)

        label = (f"{'+'.join(map(str, visits))} m "
                 f"(AUC {auc:.2f} [{lo:.2f}–{hi:.2f}])")
        plt.plot(base_fpr, mean_t, color=palette[idx], lw=1.8, label=label)

        rows_csv.append(dict(Visits="+".join(map(str, visits)),
                             AUC=auc, CI_low=lo, CI_high=hi))

    # ---- figure & table output ----------------------------------
    plt.plot([0, 1], [0, 1], "k:", lw=.8)
    plt.xlabel("False-Positive Rate"); plt.ylabel("True-Positive Rate")
    plt.title("Iterative ROC – Random Forest")
    plt.legend(fontsize=7)
    plt.tight_layout()

    cfg.plots.mkdir(parents=True, exist_ok=True)
    fig_fn = cfg.plots / f"{cfg.outcome}_iterative_ROC.png"
    plt.savefig(fig_fn, dpi=300); plt.close()

    pd.DataFrame(rows_csv).to_csv(
        cfg.tables / f"{cfg.outcome}_iterative_ROC_AUC.csv", index=False
    )
    print("iterative ROC plot  ->", fig_fn)

# ------------------------------------------------------------------ #
# 3. CLI                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oai_pickle",   required=True)
    ap.add_argument("--outcome",      default="tkr", choices=["tkr", "oa"])
    ap.add_argument("--label_window", type=int, default=48)
    ap.add_argument("--pred_window",  type=int, default=48)
    ap.add_argument("--censor_time",  type=int)
    ap.add_argument("--root",         type=Path,
                    default=Path("./exploratory_runs"))
    ap.add_argument(
        "--impute",
        action="store_true",
        help="Use mean-imputation and keep rows with missing biomarker values",
    )
    args = ap.parse_args()

    cfg = Config(label_window=args.label_window,
                 pred_window=args.pred_window,
                 censor_time=args.censor_time,
                 outcome=args.outcome,
                 root=args.root,
                 impute=args.impute)

    df = pd.read_pickle(args.oai_pickle)

    VISIT_SETS = [
        (0,),
        (0, 12),
        (0, 12, 24),
        (0, 12, 24, 36),
        (0, 12, 24, 36, 48),
        (0, 12, 24, 36, 48, 72),
        (0, 12, 24, 36, 48, 72, 96),
    ]
    run_iterative_roc(cfg, df, VISIT_SETS)
