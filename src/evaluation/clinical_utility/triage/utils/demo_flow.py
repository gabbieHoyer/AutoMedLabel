import os, ast, pandas as pd, numpy as np

def summarise_demographics(df: pd.DataFrame) -> pd.Series:
    """
    Basic demographic summary for any subject subset.

    Required columns
    ----------------
    sex          : 'M' / 'F'
    sex_num      : 0 for 'M', 1 for 'F'   
    age          : numeric (years)
    Weight[kg]   : numeric (kg)
    """
    # keep only rows with recognised sex codes
    df = df[df["sex"].isin(["M", "F"])]

    # percentage female (optional)
    pct_f = np.nan
    if "sex_num" in df.columns and df["sex_num"].notna().any():
        pct_f = np.round(df["sex_num"].mean() * 100, 1)

    out = {
        "N":            len(df),
        "sex_M":        (df["sex"] == "M").sum(),
        "sex_F":        (df["sex"] == "F").sum(),
        "sex_pct_F":    pct_f,                     # 52.7 means 52.7% female
        "age_mean":     np.round(df["age"].mean(), 1),
        "age_sd":       np.round(df["age"].std(ddof=0), 1),
        "weight_mean":  np.round(df["Weight[kg]"].mean(), 1),
        "weight_sd":    np.round(df["Weight[kg]"].std(ddof=0), 1),
    }
    return pd.Series(out)


class DemoCollector:
    """
    Collect demographic snapshots during a pipeline run and
    write a stacked summary table at the end.
    """
    def __init__(self):
        self._rows = {}

    # -------- public API ------------------------------------
    def add(self, label: str, df_subset: pd.DataFrame):
        """Store demographic stats for a named cohort."""
        self._rows[label] = summarise_demographics(df_subset)

    def final_table(self) -> pd.DataFrame:
        wanted_cols = ["N","sex_M","sex_F","sex_pct_F",
                       "age_mean","age_sd","weight_mean","weight_sd"]
        return (pd.DataFrame(self._rows)
                  .T[wanted_cols]
                  .rename_axis("Stage / Subset"))

    def save(self, cfg, fname="demographics"):
        from .io import save_table
        tbl = self.final_table()
        save_table(tbl, fname, cfg)
        return tbl

