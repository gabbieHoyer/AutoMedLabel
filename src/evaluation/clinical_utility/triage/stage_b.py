# triage/stage_b.py
import ast, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from .utils.models     import train_base_models, mean_ensemble, stack_probs
from .utils.bootstrap  import sens_at_spec, bootstrap_metric
from .utils.plotting   import roc_panel
from .utils.io         import save_table, save_fig
from .utils.plot_style import set_msk_style, cm2inch
set_msk_style()  

def run_stage_b(df, prob_a_stack, thr_a, cfg):
    """
    Parameters
    ----------
    df            : DataFrame already processed by `prepare_dataframe`
    prob_a_stack  : Series of Stage-A stack probabilities
    thr_a         : float, 90%-specificity threshold from Stage A
    cfg           : dict returned by yaml.load

    Returns
    -------
    oof_b   : dict of Series with LR / XGB / HGB / ENS / STACK scores
    thr_b   : float, 90%-specificity threshold inside A-pass knees
    tbl_b   : DataFrame with AUC, CI, sensitivity, threshold
    mask_a_pass : Boolean mask of knees passing Stage A
    """
    # ------------------------------------------------------------------
    # 1  target definition
    BC_LABELS = [
        'MedFemCart','PatellaCart','TroFemCart','MedTibCart','LatTibCart',
        'LatFemCart','PatellaBone','MedFemBone','TroFemBone','MedTibBone',
        'LatTibBone','LatFemBone'
    ]
    df["bone_cart_abnormal"] = df["multilabel"].apply(
        lambda s: int(any(lbl in ast.literal_eval(s) for lbl in BC_LABELS))
    )

    METRIC_Z = [c for c in df.columns if c.endswith("_z")]
    DEMO_Z   = ["sex_num", "age", "Weight[kg]"]

    X, y = df[METRIC_Z + DEMO_Z], df["bone_cart_abnormal"].astype(int)
    gkf  = GroupKFold(n_splits=5)

    # ------------------------------------------------------------------
    # 2  cross-validated base models
    oof = {m: pd.Series(0., index=df.index) for m in ["LR","XGB","HGB"]}
    for tr, te in gkf.split(X, y, groups=df["subject_id"]):
        lr, xgb_clf, hgb = train_base_models(X.iloc[tr], y.iloc[tr], cfg)
        oof["LR"].iloc[te]  = lr.predict_proba(X.iloc[te])[:, 1]
        oof["XGB"].iloc[te] = xgb_clf.predict_proba(
                                  X.iloc[te].to_numpy())[:, 1]
        oof["HGB"].iloc[te] = hgb.predict_proba(X.iloc[te])[:, 1]

    oof["ENS"]   = mean_ensemble(oof["LR"], oof["XGB"], oof["HGB"])
    oof["STACK"] = stack_probs([oof["LR"], oof["XGB"], oof["HGB"]], y, cfg)

    # ------------------------------------------------------------------
    # 3  threshold & bootstrap
    mask_a_pass = prob_a_stack >= thr_a
    spec_target = cfg["thresholds"]["stage_b_to_c_spec"]   # 0.85 or 0.90

    rows = []
    for name, p in oof.items():
        auc, lo_auc, hi_auc = bootstrap_metric(roc_auc_score, y, p)
        sens90, thr90       = sens_at_spec(
            y[mask_a_pass], p[mask_a_pass], spec_target
        )

        def sens_fn(yy, pp):
            return sens_at_spec(yy, pp, spec_target)[0]

        sens_mean, lo_s, hi_s = bootstrap_metric(
            sens_fn, y[mask_a_pass], p[mask_a_pass]
        )
        rows.append([name, auc, lo_auc, hi_auc,
                     sens_mean, lo_s, hi_s, thr90])

    tbl = pd.DataFrame(rows, columns=[
        "model","AUC","AUC_lo","AUC_hi",
        "sens@spec","sens_lo","sens_hi","thr_spec"
    ]).set_index("model")

    # the routing threshold uses STACK
    thr_b = tbl.loc["STACK","thr_spec"]

    # ------------------------------------------------------------------
    # 4  ROC plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for model, p in oof.items():
        ci = tbl.loc[model, ["AUC", "AUC_lo", "AUC_hi"]].values
        roc_panel(ax, y, p, model, auc_ci=ci)

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_title("Stage B ROC"); 
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("True-Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); 

    save_fig(fig, "stageB_roc", cfg,
         png=True, svg=True, svg_cm=8.9)

    # ------------------------------------------------------------------
    # 5  save & return
    save_table(tbl, "stageB_bootstrap", cfg)
    return oof, thr_b, tbl, mask_a_pass
