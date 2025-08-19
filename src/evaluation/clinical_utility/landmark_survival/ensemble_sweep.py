#!/usr/bin/env python
"""
One-off model-choice run (single landmark = 0m).

Compares LR, RF, XGB, soft-voting and stacking **using only visits that
fall inside cfg.label_window** (default 48m).  
Outputs:
  • combined ROC figure
  • CSV with mean AUC per model
  • per-model OOF predictions (parquet) - optional audit trail
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier

from landmark_pipeline import Config
from data_utils import clean_oai, get_feature_fn, BIOMETRIC_COLS, DEMOGRAPHIC_COLS
from outcomes   import get_label_fn

from style_tools import set_msk_style, roc_panel, get_colour
set_msk_style()

# ------------------------------------------------------------------#
# model zoo ---------------------------------------------------------#
def build_clf_dict(seed: int) -> Dict[str, Pipeline]:
    imputer = SimpleImputer(strategy="mean")

    lr  = make_pipeline(imputer, StandardScaler(),
                        LogisticRegression(penalty="l1", C=2.0,
                                           solver="liblinear",
                                           max_iter=1000,
                                           random_state=seed))

    rf  = make_pipeline(imputer,
                        RandomForestClassifier(n_estimators=400,
                                               min_samples_leaf=10,
                                               random_state=seed,
                                               class_weight="balanced"))

    xgb = make_pipeline(imputer,
                        XGBClassifier(n_estimators=400, max_depth=4,
                                      learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      eval_metric="logloss",
                                      random_state=seed))

    vote = VotingClassifier(estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
                            voting="soft", n_jobs=-1)

    stack = StackingClassifier(
                estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
                final_estimator=LogisticRegression(max_iter=1000,
                                                   random_state=seed),
                cv=5, n_jobs=-1, passthrough=False)

    return {"LR": lr, "RF": rf, "XGB": xgb, "Vote": vote, "Stack": stack}

# ------------------------------------------------------------------#
def run_ensemble_sweep(cfg: Config,
                       df_raw: pd.DataFrame,
                       visits: Tuple[int, ...],
                       roc_store: dict, auc_store: dict, ci_store: dict, rec_list: list
                       ) -> None:

    df        = clean_oai(df_raw, outcome=cfg.outcome, censor_time=cfg.censor_time)
    label_fn  = get_label_fn(cfg.outcome)

    feat_fn   = get_feature_fn(cfg.outcome)
    if cfg.outcome == "oa" and cfg.impute:
        # temporarily reuse the TKR builder
        feat_fn = get_feature_fn("tkr")

    # ---------- design matrix & outcome ----------------------------
    X = feat_fn(df, visits)
    y = label_fn(df, X.index, landmark_month=0,
                 horizon_months=cfg.pred_window)

    cols  = [f"{c}_m{tp}" for tp in visits for c in BIOMETRIC_COLS]
    cols += list(DEMOGRAPHIC_COLS)
    X_use = X[cols]

    # ---------- CV --------------------------------------------------
    cv        = StratifiedGroupKFold(n_splits=cfg.cv_folds,
                                     shuffle=True,
                                     random_state=cfg.cv_seed)
    base_fpr  = np.linspace(0, 1, 100)
    fig, ax   = plt.subplots(figsize=(8, 6), dpi=110)
    
    oof_dir = cfg.tables / "oof_preds"
    oof_dir.mkdir(parents=True, exist_ok=True)

    for name, clf in build_clf_dict(cfg.cv_seed).items():
        oof      = np.zeros(len(y))
        mean_tpr = np.zeros_like(base_fpr)

        for tr, te in cv.split(X_use, y, groups=X_use.index):
            clf.fit(X_use.iloc[tr], y.iloc[tr])
            prob    = clf.predict_proba(X_use.iloc[te])[:, 1]
            oof[te] = prob
            fpr, tpr, _ = roc_curve(y.iloc[te], prob)
            mean_tpr += np.interp(base_fpr, fpr, tpr)

        mean_tpr /= cv.get_n_splits()
        auc       = roc_auc_score(y, oof)

        # ---------- 1000-× bootstrap for 95 % CI -----------------
        boot = [roc_auc_score(*resample(y, oof)) for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # ----------  stash for the cross-visit figure  --------------
        roc_store[(name, visits)] = mean_tpr          # 100-pt vector
        auc_store[(name, visits)] = auc
        ci_store[(name, visits)] = (lo, hi) 
        rec_list.append({                              # row for CSV
            "Model":  name,
            "Visits": "+".join(map(str, visits)),
            "AUC":    round(auc,   3),
            "CI_low": round(lo,    3),
            "CI_high":round(hi,    3),
        })
        # ---- plotting call ----
        roc_panel(
            ax, y, oof, name,
            auc_ci=(auc, lo, hi),      # supplies mean±CI
            colour=get_colour(name),   # consistent hue across figures
        )

        pd.DataFrame({"subject_id": X_use.index,
                      "truth": y.values,
                      "pred":  oof}
                    ).to_parquet(oof_dir / f"{name}_LM0_oof.parquet")

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{cfg.outcome.upper()} ensemble ROC (≤ {cfg.label_window} m)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    (cfg.plots / f"{cfg.outcome}_ensemble_ROC.png").parent.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "svg"): #"pdf"):
        fn = cfg.plots / f"{cfg.outcome}_ensemble_ROC.{ext}"
        plt.savefig(fn, dpi=300 if ext == "png" else None)

    plt.close(fig)

    print("✓ ensemble sweep done →", cfg.tables)


# ------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oai_pickle", required=True)
    ap.add_argument("--outcome",    default="tkr", choices=["tkr", "oa"])
    ap.add_argument("--label_window", type=int, default=48)
    ap.add_argument("--pred_window",  type=int, default=48)  # match landmark run
    ap.add_argument("--censor_time",  type=int) 
    ap.add_argument("--root", type=Path, default=Path("./landmark_runs"))
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

    EXPERIMENTS = [
    (0,), (0,12), (0,12,24), (0,12,24,36), (0,12,24,36,48)
    ]
    # scratch containers that span all visit-sets  ---------------
    # roc_curves = {}      # (model, visits) -> mean-TPR vector
    # auc_vals   = {}      # (model, visits) -> scalar AUC
    roc_curves, auc_vals, ci_vals = {}, {}, {}   # <- add ci_vals
    rows_long  = []      # list of dicts for long-form CSV

    for visits in EXPERIMENTS:
        run_ensemble_sweep(cfg, df, visits,
                           roc_curves, auc_vals, ci_vals, rows_long)

    # --------  ONE wide AUC table  ---------------------------------
    df_long = pd.DataFrame(rows_long)
    df_long.to_csv(cfg.tables / f"{cfg.outcome}_ensemble_AUC_long.csv",
                   index=False)
    (df_long
        .pivot(index="Visits", columns="Model", values="AUC")
        .to_csv(cfg.tables / f"{cfg.outcome}_ensemble_AUC_pivot.csv"))

    # --------  Multi-panel ROC across visit-sets  -------------------
    VISITS_TXT = ["+".join(map(str, v)) for v in EXPERIMENTS]      # e.g. "0+12"
    visit_cols = dict(zip(
        VISITS_TXT,
        sns.color_palette("colorblind", len(EXPERIMENTS))
    ))

    MODELS    = ["LR", "RF", "XGB", "Vote", "Stack"]
    base_fpr  = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(4.2*len(MODELS), 4),
                             sharey=True)

    for ax, mdl in zip(axes, MODELS):
        for visits in EXPERIMENTS:
            key      = "+".join(map(str, visits))          # text label
            tpr      = roc_curves[(mdl, visits)]
            auc      = auc_vals [(mdl, visits)]
            lo, hi   = ci_vals [(mdl, visits)]

            ax.plot(
                base_fpr, tpr,
                lw=1.5,
                color=visit_cols[key],
                label=f"{key} m (AUC {auc:.2f} [{lo:.2f}, {hi:.2f}])",
            )

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.set_title(f"{mdl}")
        ax.set_xlabel("False-Positive Rate")
        if ax is axes[0]:
            ax.set_ylabel("True-Positive Rate")
        ax.legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    for ext in ("png", "svg"): #"pdf"):
        fn = cfg.plots / f"{cfg.outcome}_multi_ROC.{ext}"
        plt.savefig(fn, dpi=300 if ext == "png" else None)

    plt.close(fig)

    print("✓ complete multi-visit sweep →", cfg.tables, cfg.plots)



# python ensemble_sweep.py \
#        --oai_pickle  /data/msk_infocommons/Users/ghoyer/shoulder/PrecisionShoulderAI/knee_inference/OAI_landmark_analysis_0702/oa_filtered_clean.pkl \
#        --outcome     tkr \
#        --label_window 48 \
#        --pred_window 48 \
#        --censor_time 120 \


# python ensemble_sweep.py \
#    --oai_pickle  /data/msk_infocommons/Users/ghoyer/shoulder/PrecisionShoulderAI/knee_inference/OAI_landmark_analysis_0702/oa_filtered_clean.pkl \
#    --outcome     oa \
#    --label_window 48 \
#    --pred_window 48 \
#    --censor_time 120 



