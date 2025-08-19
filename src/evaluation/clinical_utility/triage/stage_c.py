# triage/stage_c.py
import ast, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from .utils.models    import fit_ensemble
from .utils.bootstrap import bootstrap_metric, sens_at_spec
from .utils.plotting  import roc_panel, calibration_ax
from .utils.io        import save_table, save_fig
from .utils.plot_style import set_msk_style, cm2inch, pretty
from .utils.plot_style import get_joint_colour, get_tissue_style

# ---------------------------------------------------------------------
# helper
sens90 = lambda y, p: sens_at_spec(y, p, 0.90)[0]

set_msk_style()  

def _evaluate(df_sub, task_list):
    rows = []
    for t in task_list:
        y, p = df_sub[t].astype(int).values, df_sub[f"{t}_prob"].values
        auc, lo, hi = bootstrap_metric(roc_auc_score, y, p)
        s , ls, hs  = bootstrap_metric(sens90,           y, p)
        rows.append([t, auc, lo, hi, s, ls, hs])
    return pd.DataFrame(rows, columns=[
        "task","AUC","AUC_lo","AUC_hi","sens90","sens_lo","sens_hi"
    ]).set_index("task")


# ---------------------------------------------------------------------
def run_stage_c(df, mask_a_pass, prob_b_stack, thr_b, cfg, *, collector=None):
    """
    Parameters
    ----------
    df            : full dataframe after preprocessing
    mask_a_pass   : Boolean mask of knees that passed Stage A
    prob_b_stack  : Series with Stage B stack probabilities
    thr_b         : routing threshold (e.g. 0.85 or 0.90 specificity)
    cfg           : experiment config

    Returns
    -------
    dict with "p1" and "p2" result tables
    """

    # ================================================================
    # 0. Initial setup for Stage‑C routed subset build
    # ================================================================
    METRIC_Z = [c for c in df.columns if c.endswith("_z")]
    DEMO_Z   = ["sex_num", "age", "Weight[kg]"]

    # ================================================================
    # 1. Part 1  (femur / tibia / patella anomaly probability)
    # ================================================================
    tasks_p1 = cfg["stage_c_tasks"]["p1"]

    # 1a  fit probability columns on *full cohort
    for t, feats_base in tasks_p1.items():
        feats = feats_base['feats'] + DEMO_Z
        df[f"{t}_prob"] = fit_ensemble(df[feats], df[t].astype(int), cfg)

    # ==== Build Stage‑C routed subset ====#
    mask_c_route = (mask_a_pass) & (prob_b_stack >= thr_b)
    df_c = df.loc[mask_c_route].copy()
    if collector is not None:
        collector.add("Stage C pass", df_c)

    # 1b  evaluate on routed knees
    perf_p1 = _evaluate(df_c, tasks_p1.keys())
    save_table(perf_p1, "stageC_p1_bootstrap", cfg)

    # 1c  ROC figure (3 curves)
    fig, ax = plt.subplots(figsize=(6,6))

    # same as before, but can drop the manual colour dict 
    for tgt in ["femur_anom", "tibia_anom", "patella_anom"]:
        ci = perf_p1.loc[tgt, ["AUC", "AUC_lo", "AUC_hi"]].values
        roc_panel(ax,
                y=df_c[tgt],
                p=df_c[f"{tgt}_prob"],
                label=pretty(tgt), 
                auc_ci=ci,
                colour=get_joint_colour(tgt))

    ax.plot([0,1],[0,1],"--",color="gray",lw=0.8)
    ax.set_title("Stage C p1 ROC"); 
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("True-Positive Rate")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout(); 
    save_fig(fig, "stageC_p1_roc", cfg,
         png=True, svg=True, svg_cm=8.9)
    
    # ---------------------------------------------------------------
    width_cm, height_cm = 17.8, 5.5        
    fig, axes = plt.subplots(
        1, 3,
        figsize=cm2inch(width_cm, height_cm),
    )

    for ax, t in zip(axes, tasks_p1):
        calibration_ax(ax, 
                       df_c[t], 
                       df_c[f"{t}_prob"], 
                       pretty(t))
    fig.tight_layout()

    save_fig(fig, "stageC_p1_calibration", cfg,
         png=True, svg=True, svg_cm=width_cm)

    # ---------- Part 1 positive groups ----------
    for tissue in tasks_p1:
        sub = df_c.loc[df_c[tissue] == 1]
        if collector is not None:
            collector.add(f"Stage C p1 – {tissue.split('_')[0]}", sub)

    # ================================================================
    # 2. Part 2  (six tissue‑biomarker tasks)
    # ================================================================
    # 2a  build additional targets if not present
    if "femur_cart" not in df.columns:
        cart_lbl = {
            'femur_cart':   ['MedFemCart','LatFemCart','TroFemCart'],
            'tibia_cart':   ['MedTibCart','LatTibCart'],
            'patella_cart': ['PatellaCart']
        }
        bone_lbl = {
            'femur_bone':   ['MedFemBone','LatFemBone','TroFemBone'],
            'tibia_bone':   ['MedTibBone','LatTibBone'],
            'patella_bone': ['PatellaBone']
        }
        def build_target(lbls):
            return df["multilabel"].apply(
                lambda s: int(any(l in ast.literal_eval(s) for l in lbls))
            )
        for name,lbls in {**cart_lbl, **bone_lbl}.items():
            df[name] = build_target(lbls)

    tasks_p2 = cfg["stage_c_tasks"]["p2"]

    # 2b  ensemble probabilities
    for t, feats_base in tasks_p2.items():
        feats = feats_base['feats'] + DEMO_Z
        df[f"{t}_prob"] = fit_ensemble(df[feats], df[t].astype(int), cfg)

    df_c = df.loc[mask_c_route].copy()

    # 2c  evaluate
    perf_p2 = _evaluate(df_c, tasks_p2.keys())
    save_table(perf_p2, "stageC_p2_bootstrap", cfg)

    # 2c.1  ROC figure (split: bone | cartilage) ─────────────────────
    bone_tasks = ["femur_bone", "tibia_bone", "patella_bone"]
    cart_tasks = ["femur_cart", "tibia_cart", "patella_cart"]

    width_cm, height_cm = 12, 6         
    fig, (ax_bone, ax_cart) = plt.subplots(
        1, 2, sharex=True, sharey=True,
        figsize=(width_cm, height_cm)
    )
    # clean legend text
    JOINT_SHORT = {"femur": "Femur", "tibia": "Tibia", "patella": "Patella"}

    # left: bone
    for task in bone_tasks:
        ci = perf_p2.loc[task, ["AUC", "AUC_lo", "AUC_hi"]].values
        joint = JOINT_SHORT[task.split("_")[0]]
        roc_panel(
            ax_bone,
            y=df_c[task],
            p=df_c[f"{task}_prob"],
            label=f"{joint}",              
            auc_ci=ci,
            colour=get_joint_colour(task),
            linestyle=get_tissue_style(task)
        )

    ax_bone.set_title("Stage C p2 ROC: Bone")
    ax_bone.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)
    ax_bone.set_xlabel("False-Positive Rate")
    ax_bone.set_ylabel("True-Positive Rate")
    ax_bone.legend(loc="lower right", fontsize=7, frameon=False)

    # right: cartilage
    for task in cart_tasks:
        ci = perf_p2.loc[task, ["AUC", "AUC_lo", "AUC_hi"]].values
        joint = JOINT_SHORT[task.split("_")[0]]
        roc_panel(
            ax_cart,
            y=df_c[task],
            p=df_c[f"{task}_prob"],
            label=f"{joint}",             
            auc_ci=ci,
            colour=get_joint_colour(task),
            linestyle=get_tissue_style(task)
        )

    ax_cart.set_title("Stage C p2 ROC: Cartilage")
    ax_cart.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)
    ax_cart.set_xlabel("False-Positive Rate")
    ax_cart.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, "stageC_p2_roc_split", cfg,
            png=True, svg=True, svg_cm=17.8) 

    # ---------------------------------------------------------------
    # 2d  calibration multi‑panel 
    width_cm, height_cm = 17.8, 10       
    fig, axes = plt.subplots(
        2, 3,
        figsize=cm2inch(width_cm, height_cm),
    )
    for ax, t in zip(axes.ravel(), tasks_p2):
        calibration_ax(ax, 
                       df_c[t], 
                       df_c[f"{t}_prob"], 
                       pretty(t))
    fig.tight_layout()

    save_fig(fig, "stageC_p2_calibration", cfg,
         png=True, svg=True, svg_cm=width_cm)

    # ---------- Part 2 positive groups ----------
    for task in tasks_p2:
        sub = df_c.loc[df_c[task] == 1]
        if collector is not None:
            collector.add(f"Stage C p2 – {task.replace('_',' ')}", sub)

    # ----------------------------------------------------------------
    # 3. Stage‑C probabilities and routing flags Export
    # ----------------------------------------------------------------
    from .utils.io import make_out_dir
    import json

    # pull out routing flags from inputs passed to run_stage_c
    # mask for knees routed to C under the current B threshold
    mask_c_route = (mask_a_pass) & (prob_b_stack >= thr_b)

    cols_id = ["subject_id"]  # "knee_id",
    cols_gt = ["femur_anom","tibia_anom","patella_anom",
            "femur_bone","tibia_bone","patella_bone",
            "femur_cart","tibia_cart","patella_cart"]
    cols_prob = [f"{c}_prob" for c in cols_gt]

    to_save = df[cols_id + cols_gt + cols_prob].copy()
    to_save["pass_A"]     = mask_a_pass.astype(int)
    to_save["prob_A"]     = None  
    to_save["prob_B"]     = prob_b_stack 
    to_save["forward_B85"] = (prob_b_stack >= 0.893).astype(int)  
    to_save["forward_B90"] = (prob_b_stack >= 0.911).astype(int)

    out_dir = make_out_dir(cfg)
    to_save.to_csv(out_dir / "triage_stageC_predictions.csv", index=False)

    # ----------------------------------------------------------------
            
    return {"p1": perf_p1, "p2": perf_p2}




















        # sharex='col',                       # share X within each column
        # sharey='row'                        # share Y within each row

    # width_cm, height_cm = 12, 6          # modestly wider than before
    # fig, (ax_bone, ax_cart) = plt.subplots(
    #     1, 2, sharex=True, sharey=True,
    #     figsize=cm2inch(width_cm, height_cm)
    # )

    # # 2‑c.1  ROC figure (6 curves)  ──────────────────────────────────
    # fig, ax = plt.subplots(figsize=(6, 6))
    # for task in tasks_p2:
    #     ci = perf_p2.loc[task, ["AUC", "AUC_lo", "AUC_hi"]].values
    #     roc_panel(
    #         ax,
    #         y=df_c[task],
    #         p=df_c[f"{task}_prob"],
    #         label=pretty(task),
    #         auc_ci=ci,
    #         colour=get_joint_colour(task),
    #         linestyle=get_tissue_style(task),
    #     )

    # ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    # ax.set_title("Stage C p2 ROC")
    # ax.set_xlabel("False‑Positive Rate"); ax.set_ylabel("True‑Positive Rate")
    # ax.legend(loc="lower right", fontsize=7, frameon=False)
    # save_fig(fig, "stageC_p2_roc", cfg, png=True, svg=True, svg_cm=8.9)


    # # left: bone
    # for task in bone_tasks:
    #     ci = perf_p2.loc[task, ["AUC", "AUC_lo", "AUC_hi"]].values
    #     roc_panel(
    #         ax_bone,
    #         y=df_c[task],
    #         p=df_c[f"{task}_prob"],
    #         label=pretty(task),
    #         auc_ci=ci,
    #         colour=get_joint_colour(task),
    #         linestyle=get_tissue_style(task),   # solid
    #     )

    # ax_bone.set_title("Bone")
    # ax_bone.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)

    # # right: cartilage
    # for task in cart_tasks:
    #     ci = perf_p2.loc[task, ["AUC", "AUC_lo", "AUC_hi"]].values
    #     roc_panel(
    #         ax_cart,
    #         y=df_c[task],
    #         p=df_c[f"{task}_prob"],
    #         label=pretty(task),
    #         auc_ci=ci,
    #         colour=get_joint_colour(task),
    #         linestyle=get_tissue_style(task),   # dashed
    #     )

    # ax_cart.set_title("Cartilage")
    # ax_cart.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)

    # # shared axis labels only once
    # for ax in (ax_bone, ax_cart):
    #     ax.set_xlabel("False-Positive Rate")
    # ax_bone.set_ylabel("True-Positive Rate")

    # # one legend, anchored below both axes
    # handles, labels = ax_bone.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center",
    #         bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7, frameon=False)



# COLORS_C1 = {"femur_anom":   "#0072B2",
#              "tibia_anom":   "#D55E00",
#              "patella_anom": "#009E73"}


    # 1‑d  calibration figure (1×3)
    # fig, axes = plt.subplots(1,3,figsize=(9,3))
    # fig, axes = plt.subplots(1,3,figsize=(9/2.54,3/2.54))



    # width_cm, height_cm = 17.8, 10          # wide, moderate height
    # fig, ax = plt.subplots(figsize=cm2inch(width_cm, height_cm / 2))
    # fig, ax = plt.subplots(figsize=(6,6))

    # for tgt in tasks_p2:                    # order from YAML
    #     ci = perf_p2.loc[tgt, ["AUC", "AUC_lo", "AUC_hi"]].values
    #     roc_panel(
    #         ax,
    #         y=df_c[tgt],
    #         p=df_c[f"{tgt}_prob"],
    #         label=pretty(tgt),               # nice label via plot_style.py
    #         auc_ci=ci
    #     )

    # ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    # ax.set_xlabel("False‑Positive Rate")
    # ax.set_ylabel("True‑Positive Rate")
    # ax.set_title("Stage C p2 ROC")
    # ax.legend(loc="lower right", fontsize=7)
    # # ax.legend(loc="lower right", fontsize=7, ncol=2)  # 2 columns keeps it tidy
    # fig.tight_layout()

    # save_fig(fig, "stageC_p2_roc", cfg,
    #          png=True, svg=True, svg_cm=width_cm)
    
    # save_fig(fig, "stageC_p2_roc", cfg,
    #      png=True, svg=True, svg_cm=8.9)

    # fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))   # 6 cm square

    # for t, col in COLORS_C1.items():
    #     # roc_panel(ax, df_c[t], df_c[f"{t}_prob"], t.replace("_", " "), col)
    #     ci = perf_p1.loc[t, ["AUC", "AUC_lo", "AUC_hi"]].values
    #     roc_panel(ax, df_c[t], df_c[f"{t}_prob"], t.replace("_"," "), ci)
    #     # roc_panel(ax, df_c[t], df_c[f"{t}_prob"], t.replace("_"," "), COLORS_C1[t], ci)


        # save_fig(fig, "stageC_p1_calibration", cfg)
    # save_fig(fig, "stageC_p1_calibration", cfg,
    #      png=True, svg=True, svg_cm=8.9)


        # save_fig(fig, "stageC_p2_calibration", cfg)
    # save_fig(fig, "stageC_p2_calibration", cfg,
    #      png=True, svg=True, svg_cm=8.9)


        # fig, axes = plt.subplots(2,3,figsize=(9,6))
    # fig, axes = plt.subplots(2,3,figsize=(9/2.54,6/2.54))
