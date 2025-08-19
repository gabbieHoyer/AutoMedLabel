"""
Shared low-level helpers: constants, cleaning, feature engineering.
Nothing here is outcome-specific.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
from typing import Sequence, Iterable, Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# 1. Canonical column sets & time-grid                               #
# ------------------------------------------------------------------ #
BIOMETRIC_COLS: Tuple[str, ...] = (
    "femoral cartilage",
    "lateral tibial cartilage",
    "medial tibial cartilage",
    "patellar cartilage",
    "lateral meniscus",
    "medial meniscus",
)
DEMOGRAPHIC_COLS: Tuple[str, ...] = ("sex", "age", "BMI")

# visits actually present in OAI 
TIMEPOINTS_ALL: Tuple[int, ...] = (0, 12, 24, 36, 48, 72, 96)

# ------------------------------------------------------------------ #
# 2. Task-based cleaning                                             #
# ------------------------------------------------------------------ #
# -------------------------- shared constants --------------------------
TKR_COLS = {"flag": "tkr",
            "time": "time_of_event_TKR",
            "out":  "first_event_time_TKR"}

OA_COLS  = {"kl":   "pred_kl",               # KL grade per visit
            "out":  "first_event_time_OA"}
    
# --------------------------------------------------------------------- #
#  a) generic fixes that apply to any outcome                           #
# --------------------------------------------------------------------- #
def _generic_clean(df: pd.DataFrame, censor_time: int) -> pd.DataFrame:
    df = df.copy()

    # add more "always-needed" fixes here (sex coding, race NA, …)
    if "tkr" in df.columns:
        df["tkr"] = df["tkr"].fillna(0).astype(int)

    return df

# --------------------------------------------------------------------- #
#  b) outcome-specific enrichers                                        #
# --------------------------------------------------------------------- #
def _ensure_tkr_columns(df: pd.DataFrame, censor_time: int) -> pd.DataFrame:
    """Adds 'first_event_time_TKR' if it doesn't exist yet."""
    if TKR_COLS["time"] not in df.columns:
        raise ValueError("time_of_event_TKR column missing from dataframe")

    if TKR_COLS["out"] not in df.columns:
        df[TKR_COLS["time"]] = df[TKR_COLS["time"]].fillna(censor_time)
        df[TKR_COLS["out"]]  = (
            df.groupby("subject_id")[TKR_COLS["time"]].transform("first")
        )
    return df


def _ensure_oa_columns(df: pd.DataFrame, censor_time: int) -> pd.DataFrame:
    """Adds 'first_event_time_OA' (first KL≥2) if absent."""
    if OA_COLS["out"] in df.columns:                     # already there
        return df

    first_evt = (
        df[df[OA_COLS["kl"]] >= 2]
          .groupby("subject_id")["months"]
          .min()
          .rename(OA_COLS["out"])
    )
    df = df.merge(first_evt, left_on="subject_id", right_index=True, how="left")
    df[OA_COLS["out"]] = df[OA_COLS["out"]].fillna(censor_time)
    return df


_CLEAN_DISPATCH = {
    "tkr": _ensure_tkr_columns,
    "oa":  _ensure_oa_columns,
    # add more outcomes later
}

# --------------------------------------------------------------------- #
#  c) one public wrapper – call this everywhere                         #
# --------------------------------------------------------------------- #
def clean_oai(
    df_raw:      pd.DataFrame,
    outcome:     str,
    censor_time: int,
) -> pd.DataFrame:
    """
    • Performs generic NA / type fixes and
    • adds the outcome-specific 'first_event_time_*' column required by
      the label helpers in 'outcomes.py'.

    Parameters
    ----------
    outcome : {"tkr", "oa"}
        Decides which additional columns are created.
    censor_time : int (months)
        Administrative censoring horizon (e.g. 120 m or 144 m).
    """
    df = _generic_clean(df_raw, censor_time)

    try:
        df = _CLEAN_DISPATCH[outcome.lower()](df, censor_time)
    except KeyError:
        raise ValueError(f"Unknown outcome '{outcome}'. "
                         f"Expected one of {list(_CLEAN_DISPATCH)}")

    return df


# ------------------------------------------------------------------ #
# 3. Feature matrix builder (works for any outcome)                  #
# ------------------------------------------------------------------ #

def make_feature_matrix(
    df: pd.DataFrame,
    visits: Iterable[int],
    *,
    biometric_cols: Optional[Sequence[str]] = None,
    demo_cols:     Optional[Sequence[str]] = None,
    impute:       bool   = True,
    drop_na:      bool   = False,
) -> pd.DataFrame:
    """
    Build a subject-indexed feature table for any list of visits.
    
    Args:
      df:            raw OAI dataframe, must include 'months' and 'subject_id'  
      visits:        timepoints to pull (e.g. [0,12,24,…])  
      biometric_cols: list of numeric cols to grab at each visit  
      demo_cols:     list of baseline demographic cols  
      impute:        if True, per-visit mean-impute; else leave NaNs  
      drop_na:       if True, drop any row with missing biometrics after merge  

    Returns:
      DataFrame indexed by subject_id, columns like
      'femoral cartilage_m0', …, plus baseline demo_cols.
    """
    bio = biometric_cols or BIOMETRIC_COLS
    demo = demo_cols      or DEMOGRAPHIC_COLS

    parts = []
    for tp in visits:
        cols = ["subject_id", *bio]
        tmp = (df[df["months"] == tp][cols]
               .set_index("subject_id"))
        if impute:
            tmp = tmp.fillna(tmp.mean())
        tmp.columns = [f"{c}_m{tp}" for c in tmp.columns]
        parts.append(tmp)

    X = pd.concat(parts, axis=1).reset_index()

    # add baseline demographics
    base = (df[df["months"] == 0]
            .drop_duplicates("subject_id")
            .loc[:, ["subject_id", *demo]])
    X = X.merge(base, on="subject_id", how="left")

    if drop_na:
        X = X.dropna()

    return X.set_index("subject_id")

# ------------------------------------------------------------------ #
# 4. Outcome-specific feature-matrix builders                        #
# ------------------------------------------------------------------ #

def make_feature_matrix_demo_only(
    df: pd.DataFrame,
    visits: Iterable[int],
    *,
    outcome: str,
    impute: bool = False,
) -> pd.DataFrame:
    """
    Build demo-only X using same subject index as outcome-specific
    builder would, so cohort matches main run.

    call the outcome builder (may drop rows differently for
    TKR vs OA), then reindex baseline demographics to that subject list.
    """
    # choose same feature fn used in main pipeline
    feat_fn = get_feature_fn(outcome)
    if outcome.lower() == "oa" and impute:
        # mirror existing fallback to TKR builder for OA+impute
        feat_fn = get_feature_fn("tkr")

    # defines exact cohort for the given visits/LM
    X_idx = feat_fn(df, visits).index

    # baseline demographics from month 0 (matches make_feature_matrix)
    base = (
        df[df["months"] == 0]
        .drop_duplicates("subject_id")
        .set_index("subject_id")[list(DEMOGRAPHIC_COLS)]
    )

    # align to cohort and ordering picked by feat_fn
    return base.reindex(X_idx)


def make_feature_matrix_tkr(
    df: pd.DataFrame,
    visits: Iterable[int],
) -> pd.DataFrame:
    """Per-visit mean-impute, keep all knees (for TKR)."""
    return make_feature_matrix(
        df,
        visits,
        biometric_cols=BIOMETRIC_COLS,
        demo_cols=DEMOGRAPHIC_COLS,
        impute=True,
        drop_na=False,
    )

def make_feature_matrix_oa(
    df: pd.DataFrame,
    visits: Iterable[int],
) -> pd.DataFrame:
    """Require complete data, no imputation (for OA incidence)."""
    return make_feature_matrix(
        df,
        visits,
        biometric_cols=BIOMETRIC_COLS,
        demo_cols=DEMOGRAPHIC_COLS,
        impute=False,
        drop_na=True,
    )

FEATURE_MATRIX_FUNCTIONS = {
    "tkr": make_feature_matrix_tkr,
    "oa":  make_feature_matrix_oa,
    # add more outcomes here
}

def get_feature_fn(outcome: str):
    try:
        return FEATURE_MATRIX_FUNCTIONS[outcome.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown outcome '{outcome}'. "
            f"Choose one of {list(FEATURE_MATRIX_FUNCTIONS)}"
        )


# --------------------------------------------------------------- #
#  registries so pipeline can stay outcome-agnostic               #
# --------------------------------------------------------------- #

# 1)  Which column holds 'first' event time?
_EVENT_TIME_COL = {"tkr": "first_event_time_TKR",
                   "oa" : "first_event_time_OA"}

def get_event_time_col(outcome: str) -> str:
    try:
        return _EVENT_TIME_COL[outcome.lower()]
    except KeyError:
        raise ValueError(f"Unknown outcome '{outcome}'. "
                         f"Expected one of {_EVENT_TIME_COL}")
        

# 2)  FP-penalty scenarios to plot in decision-curves
#     can tweak dictionary once and every script uses.
_DC_SCENARIOS = {
    "tkr": {                         # name -> FP-weight
        "Standard_FP": 0.0,
        "MRI_FP0.2"  : 0.20,
        "Surg_FP1.0" : 1.00,
    },
    "oa": {
        "Standard_FP"      : 0.0,
        "MRI_FP0.2"        : 0.20,
        "PrevCounsel_FP1.0": 1.00,   # <- demo counselling example
    },
}

def get_dc_scenarios(outcome: str) -> dict[str, float]:
    try:
        return _DC_SCENARIOS[outcome.lower()]
    except KeyError:
        raise ValueError(f"Unknown outcome '{outcome}'. "
                         f"Expected one of {_DC_SCENARIOS}")

