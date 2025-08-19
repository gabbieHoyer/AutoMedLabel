# model_utils.py
from typing import Dict
from sklearn.pipeline      import make_pipeline
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from xgboost               import XGBClassifier         


def get_lr(seed: int = 0):
    """Median-imputed, L2-regularised, class-balanced logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed)
    )


def get_rf(seed: int = 0):
    """Median-imputed, class-balanced random forest."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed)
    )


def get_xgb(seed: int = 0):
    """helper so XGB uses the same imputer."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed)
    )


def get_model_dict(seed: int = 0) -> Dict[str, object]:
    """Handy if need the full zoo in one call."""
    return {"LR": get_lr(seed), "RF": get_rf(seed), "XGB": get_xgb(seed)}
