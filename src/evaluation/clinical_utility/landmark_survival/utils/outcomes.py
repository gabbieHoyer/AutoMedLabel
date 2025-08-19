"""
Outcome-specific labelling helpers.
Each helper returns a binary Series (index = knees) that matches the
index passed in.

You can later add more (revision surgery, re-tear, etc.) without ever
touching the landmark code.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# 1. TKR within horizon                                              #
# ------------------------------------------------------------------ #
def label_events_tkr(
    df: pd.DataFrame,
    at_risk_idx: pd.Index,
    landmark_month: int,
    horizon_months: int,
) -> pd.Series:
    """
    Definition: “Any total/partial knee replacement between LM and
    LM+horizon (inclusive).”
    Assumes `first_event_time_TKR` is already in the dataframe
    (see `data_utils.clean_oai`).
    """
    first_time = (
        df.drop_duplicates("subject_id")
          .set_index("subject_id")
          .loc[at_risk_idx, "first_event_time_TKR"]
    )
    return (first_time <= landmark_month + horizon_months).astype(int)


# ------------------------------------------------------------------ #
# 2. Incident OA (KL≥2) within horizon                               #
# ------------------------------------------------------------------ #
def add_oa_first_event_time(df: pd.DataFrame, censor_time: int) -> pd.DataFrame:
    """
    Adds column `first_event_time_OA` (months to first KL≥2).
    Call **once** in your project-level cleaning script.
    """
    first_evt = (
        df[df["pred_kl"] >= 2]
          .groupby("subject_id")["months"]
          .min()
          .rename("first_event_time_OA")
    )
    df = df.merge(first_evt, left_on="subject_id", right_index=True, how="left")
    df["first_event_time_OA"] = df["first_event_time_OA"].fillna(censor_time)
    return df


def label_events_oa(
    df: pd.DataFrame,
    at_risk_idx: pd.Index,
    landmark_month: int,
    horizon_months: int,
) -> pd.Series:
    first_time = (
        df.drop_duplicates("subject_id")
          .set_index("subject_id")
          .loc[at_risk_idx, "first_event_time_OA"]
    )
    return (first_time <= landmark_month + horizon_months).astype(int)


# ------------------------------------------------------------------ #
# 3. dispatcher so landmark_pipeline can stay agnostic               #
# ------------------------------------------------------------------ #
LABEL_FUNCTIONS = {
    "tkr": label_events_tkr,
    "oa" : label_events_oa,
    # "revision": label_events_revision,  <- add later …
}

def get_label_fn(outcome: str):
    try:
        return LABEL_FUNCTIONS[outcome.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown outcome '{outcome}'. "
                         f"Available: {list(LABEL_FUNCTIONS)}") from e
