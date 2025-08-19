#!/usr/bin/env python
"""
Baseline-only Cox PH benchmark (month-0 biomarkers + demographics).
Outputs:
  - fold-wise C-index CSV
  - baseline survival curve
  - KM curves by median risk
All results land in cfg.tables / cfg.plots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index  # plain Harrell’s C

# IPCW-adjusted C comes from scikit-survival
from sksurv.metrics import concordance_index_ipcw
from sksurv.util     import Surv                   # helper to build structured arrays

from landmark_pipeline import Config
from data_utils import clean_oai, get_feature_fn, BIOMETRIC_COLS, DEMOGRAPHIC_COLS
from outcomes     import get_label_fn

# ------------------------------------------------------------------#
def run_survival_baseline(cfg: Config, df_raw: pd.DataFrame) -> None:

    df       = clean_oai(df_raw, outcome=cfg.outcome, censor_time=cfg.censor_time)
    label_fn = get_label_fn(cfg.outcome)

    feat_fn  = get_feature_fn(cfg.outcome)
    
    if cfg.outcome == "oa" and cfg.impute:
        # temporarily reuse the TKR builder
        feat_fn = get_feature_fn("tkr")

    # -------- design matrix (baseline only) ------------------------
    X0        = feat_fn(df, [0])
    feat_cols = [f"{c}_m0" for c in BIOMETRIC_COLS] + list(DEMOGRAPHIC_COLS)

    # -------- duration & event flags -------------------------------
    duration = df.drop_duplicates("subject_id").set_index("subject_id")[
        f"first_event_time_{cfg.outcome.upper()}"
    ].reindex(X0.index).fillna(cfg.censor_time)

    # “did the event happen within the forecast horizon?”
    event = label_fn(df,
                    X0.index,
                    landmark_month=0,
                    horizon_months=cfg.pred_window)


    surv_df = pd.concat([duration.rename("duration"),
                         event.rename("event"),
                         X0[feat_cols]], axis=1)

    # -------- cross-validated C-index ------------------------------
    cv = StratifiedGroupKFold(n_splits=cfg.cv_folds,
                              shuffle=True, random_state=cfg.cv_seed)
    ci_raw  = []
    ci_ipcw = []

    for tr, te in cv.split(surv_df[feat_cols],
                           surv_df["event"],
                           groups=surv_df.index):
        
        cph   = CoxPHFitter().fit(surv_df.iloc[tr], "duration", "event")
        risk  = cph.predict_partial_hazard(surv_df.iloc[te]).values.ravel()

        # ---------- plain C ----------------------------------------
        ci_plain = concordance_index(
            surv_df.iloc[te]["duration"],
            -risk,
            surv_df.iloc[te]["event"],
        )
        ci_raw.append(ci_plain)

        # ---------- IPCW C -----------------------------------------
        y_train = Surv.from_arrays(
            event=surv_df.iloc[tr]["event"].astype(bool).values,
            time =surv_df.iloc[tr]["duration"].values,
        )
        y_test  = Surv.from_arrays(
            event=surv_df.iloc[te]["event"].astype(bool).values,
            time =surv_df.iloc[te]["duration"].values,
        )
        # risk should not be negated for lifelines IPCW
        ci_ipcw_val = concordance_index_ipcw(
                 y_train, y_test, risk, tau=cfg.pred_window)[0]
        
        ci_ipcw.append(ci_ipcw_val)

    # print(f"C-index {np.mean(ci_scores):.3f} ± {np.std(ci_scores):.3f}")
    print(f"C-index (naïve)   : {np.mean(ci_raw):.3f} ± {np.std(ci_raw):.3f}")
    print(f"C-index (IPCW)    : {np.mean(ci_ipcw):.3f} ± {np.std(ci_ipcw):.3f}")
 
    # -------- write outputs ----------------------------------------
    cfg.tables.mkdir(parents=True, exist_ok=True)
    cfg.plots .mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "fold":     range(1, len(ci_raw)+1),
        "c_index":  ci_raw,
        "c_index_ipcw": ci_ipcw}
        ).to_csv(cfg.tables / f"{cfg.outcome}_cox_cindex.csv",
                    index=False)
    
    # -------- summary row (mean ± SD) ------------------------------
    summary_df = pd.DataFrame({
        "metric": ["c_index", "c_index_ipcw"],
        "mean":   [np.mean(ci_raw),  np.mean(ci_ipcw)],
        "sd":     [np.std(ci_raw),   np.std(ci_ipcw)],
    })
    summary_df.to_csv(cfg.tables / f"{cfg.outcome}_cox_summary.csv",
                    index=False)

    # full-data model
    cph_full = CoxPHFitter().fit(surv_df, "duration", "event")
    cph_full.baseline_survival_.plot(figsize=(6,4))
    plt.title("Baseline survival"); plt.xlabel("Months"); plt.ylabel("S(t)")
    plt.tight_layout()

    for ext in ("png", "svg"): #"pdf"):
        fn = cfg.plots / f"{cfg.outcome}_baseline_survival.{ext}"
        plt.savefig(fn, dpi=300 if ext == "png" else None)

    plt.close()

    # KM by median risk
    risk = cph_full.predict_partial_hazard(surv_df)
    high = risk >= risk.median()
    kmf  = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for lab, mask in [("Low risk", ~high), ("High risk", high)]:
        kmf.fit(surv_df.loc[mask, "duration"],
                surv_df.loc[mask, "event"], label=lab)
        kmf.plot_survival_function(ci_show=False)
    plt.title("Kaplan-Meier by risk stratum")
    plt.xlabel("Months"); plt.ylabel("S(t)")
    plt.tight_layout()

    for ext in ("png", "svg"): #"pdf"):
        fn = cfg.plots / f"{cfg.outcome}_KM_risk.{ext}"
        plt.savefig(fn, dpi=300 if ext == "png" else None)

    plt.close()
    print("✓ survival baseline outputs →", cfg.plots)

# ------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oai_pickle", required=True)
    ap.add_argument("--outcome",    default="tkr", choices=["tkr", "oa"])
    ap.add_argument("--label_window", type=int, default=48)
    ap.add_argument("--pred_window",  type=int, default=48)  # match landmark run
    ap.add_argument("--censor_time",  type=int) 
    ap.add_argument("--root",       type=Path, default=Path("./landmark_runs"))
    ap.add_argument(
        "--impute",
        action="store_true",
        help="Use mean-imputation and keep rows with missing biomarker values",
    )
    args = ap.parse_args()

    # cfg = Config(outcome=args.outcome, root=args.root)
    cfg = Config(label_window=args.label_window,
                pred_window=args.pred_window,
                censor_time=args.censor_time,
                outcome=args.outcome,
                root=args.root,
                impute=args.impute)
    
    df  = pd.read_pickle(args.oai_pickle)
    run_survival_baseline(cfg, df)


