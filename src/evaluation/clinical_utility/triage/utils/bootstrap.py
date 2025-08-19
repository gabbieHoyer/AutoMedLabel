import os, ast, pandas as pd, numpy as np
from numpy.random import default_rng
from sklearn.metrics import (
    roc_curve, roc_auc_score, auc as sk_auc
)
from functools import partial
from operator import itemgetter

def sens_at_spec(y, p, spec_target=0.90):
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    idx  = np.argmin(np.abs(spec - spec_target))
    return tpr[idx], thr[idx]

def bootstrap_metric(func, y, p, n=2000, seed=42):
    """
    Bootstraps 'func(y_true, y_pred)' and returns
    (mean, 2.5%-tile, 97.5%-tile).

    Any resample that ends up with a single class is ignored so
    metrics such as ROC-AUC remain well-defined.
    """
    rng    = default_rng(seed)
    y_arr  = np.asarray(y)
    p_arr  = np.asarray(p)
    vals   = []

    while len(vals) < n:                     # keep going until have n valid reps
        idx      = rng.integers(0, len(y_arr), len(y_arr))
        y_sample = y_arr[idx]
        if np.unique(y_sample).size < 2:     # skip single-class draw
            continue
        vals.append(func(y_sample, p_arr[idx]))

    vals = np.asarray(vals)
    mean, lo, hi = vals.mean(), *np.percentile(vals, [2.5, 97.5])
    return mean, lo, hi


sens90_fn = lambda y, p: itemgetter(0)(partial(sens_at_spec, spec_target=0.90)(y, p))
