import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from .utils.models import train_base_models, mean_ensemble, stack_probs
from .utils.bootstrap import bootstrap_metric, sens_at_spec
from .utils.plotting import roc_panel
from .utils.io import save_table, save_fig
from .utils.plot_style import set_msk_style, cm2inch

set_msk_style()  

def run_stage_a(df, cfg):
    METRIC_Z = [c for c in df.columns if c.endswith("_z")]
    DEMO_Z   = ["sex_num", "age", "Weight[kg]"]
    PLOT_ALL = cfg.get("stage_a_plot_all", False)  
    
    spec_target = cfg["thresholds"]["stage_a_spec"]   # 0.90 by default

    df["any_abnormal"] = df["multilabel"].apply(
    lambda s: int(ast.literal_eval(s) != ["normal"])
    )

    X, y = df[METRIC_Z + DEMO_Z], df["any_abnormal"].astype(int)
    gkf = GroupKFold(n_splits=5)

    oof = {m: pd.Series(0., index=df.index) for m in ["LR","XGB","HGB"]}
    for tr, te in gkf.split(X, y, groups=df["subject_id"]):
        lr, xgb_clf, hgb = train_base_models(X.iloc[tr], y.iloc[tr], cfg)
        oof["LR"].iloc[te]  = lr.predict_proba(X.iloc[te])[:,1]
        oof["XGB"].iloc[te] = xgb_clf.predict_proba(X.iloc[te].to_numpy())[:,1]
        oof["HGB"].iloc[te] = hgb.predict_proba(X.iloc[te])[:,1]

    oof["ENS"]   = mean_ensemble(oof["LR"], oof["XGB"], oof["HGB"])
    oof["STACK"] = stack_probs([oof["LR"], oof["XGB"], oof["HGB"]], y, cfg)

    rows = []
    for name, p in oof.items():

        # point estimates on full data
        auc_point               = roc_auc_score(y, p)
        sens_point, thr_spec    = sens_at_spec(y, p, spec_target)

        # bootstrap confidence limits
        _, auc_lo,  auc_hi      = bootstrap_metric(roc_auc_score, y, p)

        def sens_fn(yy, pp):
            return sens_at_spec(yy, pp, spec_target)[0]

        _, sens_lo, sens_hi     = bootstrap_metric(sens_fn, y, p)

        rows.append([
            name,
            auc_point, auc_lo, auc_hi,
            sens_point, sens_lo, sens_hi,
            thr_spec
        ])

    tbl = pd.DataFrame(
        rows,
        columns=[
            "model", "AUC", "AUC_lo", "AUC_hi",
            f"sens@{int(spec_target*100)}", "sens_lo", "sens_hi",
            f"thr_{int(spec_target*100)}"
        ]
    ).set_index("model")

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    models_to_show = ["LR", "XGB", "HGB", "ENS", "STACK"] if PLOT_ALL else ["LR", "STACK"]
    for m in models_to_show:
        ci = tbl.loc[m, ["AUC","AUC_lo","AUC_hi"]].values
        roc_panel(ax, y, oof[m], m, auc_ci=ci)

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_title("Stage A ROC");
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("True-Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); 

    save_fig(fig, "stageA_roc", cfg,
         png=True, svg=True, svg_cm=8.9)

    save_table(tbl, "stageA_bootstrap", cfg)
    thr = tbl.loc["STACK", f"thr_{int(spec_target*100)}"]
    return oof, thr, tbl




