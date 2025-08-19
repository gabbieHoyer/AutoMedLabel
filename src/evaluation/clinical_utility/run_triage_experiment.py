import argparse, yaml, json
import pandas as pd
from sklearn.metrics import confusion_matrix

# from triage.utils.io import load_csv, make_out_dir
# from triage.utils.preprocessing import prepare_dataframe
# from triage.utils.bootstrap import sens_at_spec
# from triage.stage_a import run_stage_a
# from triage.stage_b import run_stage_b  
# from triage.stage_c import run_stage_c 
# from triage.utils.demo_flow import DemoCollector, summarise_demographics

from clinical_utility.triage import (
    run_stage_a, run_stage_b, run_stage_c,
    load_csv, make_out_dir, prepare_dataframe,
    sens_at_spec, DemoCollector, summarise_demographics,
)

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--tag", default=None)
    return p.parse_args()

def main():
    args = parse_cli()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["tag"] = args.tag

    collector = DemoCollector()

    df_raw = load_csv(cfg)

    df = prepare_dataframe(df_raw, cfg)

    # ─── full cohort ──────────────────────────────────────────
    collector.add("Full cohort", df)

    # ─── Stage A ──────────────────────────────────────────────
    oof_a, thr_a, tbl_a = run_stage_a(df, cfg)
    mask_a_pass = oof_a["STACK"] >= thr_a
    collector.add("Stage A pass", df.loc[mask_a_pass])

    # ─── Stage B ──────────────────────────────────────────────
    oof_b, thr_b, tbl_b, mask_a_pass = run_stage_b(
        df,               # full dataframe
        oof_a["STACK"],   # probabilities
        thr_a,            # threshold from Stage A
        cfg
    )
    df_b = df.loc[mask_a_pass]          
    collector.add("Stage B eval set", df_b)

    # ─── Stage C ──────────────────────────────────────────────
    results_c = run_stage_c(
        df, mask_a_pass, oof_b["STACK"], thr_b, cfg,
        collector=collector
    )

    # --- export knee-level predictions for counts ---
    out_dir = make_out_dir(cfg)

    # thresholds already used for A; derive both B85 and B90 from oof_b inside A-pass
    thr_B85 = sens_at_spec(df["bone_cart_abnormal"][mask_a_pass],
                        oof_b["STACK"][mask_a_pass], 0.85)[1]
    thr_B90 = sens_at_spec(df["bone_cart_abnormal"][mask_a_pass],
                        oof_b["STACK"][mask_a_pass], 0.90)[1]

    # choose an ID for each knee
    knee_id_col = "study_id" if "study_id" in df.columns else None
    ids = df[knee_id_col] if knee_id_col else df.index.to_series().rename("knee_idx")

    pred_df = pd.DataFrame({
        "knee_id": ids,
        "subject_id": df["subject_id"],
        "y_true_knee_abnormal": df["bone_cart_abnormal"].astype(int),
        "pred_A": oof_a["STACK"].astype(float),
        "pred_B": oof_b["STACK"].astype(float)
    })

    pred_df["pass_A"]    = pred_df["pred_A"] >= float(thr_a)
    pred_df["forward_B85"] = pred_df["pass_A"] & (pred_df["pred_B"] >= float(thr_B85))
    pred_df["forward_B90"] = pred_df["pass_A"] & (pred_df["pred_B"] >= float(thr_B90))

    pred_csv = out_dir / "triage_knee_level_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    def cm_counts(y_true, forwarded):
        tn, fp, fn, tp = confusion_matrix(y_true.astype(int), forwarded.astype(int)).ravel()
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        return tn, fp, fn, tp, npv, ppv

    rows = []
    for tag, mask in [("B85", pred_df["forward_B85"]), ("B90", pred_df["forward_B90"])]:
        tn, fp, fn, tp, npv, ppv = cm_counts(pred_df["y_true_knee_abnormal"], mask)
        rows.append({
            "setting": tag,
            "forwarded": int(mask.sum()),
            "TN_removed_normals": int(tn),
            "FP_forwarded_normals": int(fp),
            "FN_missed_abnormals": int(fn),
            "TP_forwarded_abnormals": int(tp),
            "NPV": float(npv),
            "PPV": float(ppv),
            "N_norm": int((pred_df["y_true_knee_abnormal"] == 0).sum()),
            "N_abn": int((pred_df["y_true_knee_abnormal"] == 1).sum())
        })

    counts_df = pd.DataFrame(rows)
    counts_csv = out_dir / "triage_cascade_counts.csv"
    counts_df.to_csv(counts_csv, index=False)

    print(f"\n✓ saved {pred_csv.name} and {counts_csv.name} in {out_dir}")
    print(counts_df)

    # ─── write demographics CSV ──────────────────────────────
    demo_tbl = collector.save(cfg, "demographics_summary")
    print("\nDemographic table\n", demo_tbl.to_string(float_format="%.1f"))

    manifest = {"thr_a": float(thr_a), "thr_b": float(thr_b)}
    out = make_out_dir(cfg) / "manifest.json"
    json.dump(manifest, open(out,"w"), indent=2)
    print(f"✓ experiment complete → {out}")

if __name__ == "__main__":
    main()


# # From one directory above clinical_utility/
# python -m clinical_utility.run_triage_experiment
