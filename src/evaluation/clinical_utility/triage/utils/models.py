import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def train_base_models(X, y, cfg):
    params = cfg["model"]["xgb"]
    lr = make_pipeline(SimpleImputer(strategy="median"),
                       StandardScaler(),
                       LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                          solver="saga", max_iter=5000,
                                          class_weight="balanced",
                                          random_state=cfg["seed"]))
    xgb_clf = xgb.XGBClassifier(**params,
                                scale_pos_weight=(y == 0).sum()/(y == 1).sum(),
                                eval_metric="logloss",
                                random_state=cfg["seed"])
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                         max_iter=250, random_state=cfg["seed"])
    lr.fit(X, y)
    xgb_clf.fit(X.to_numpy(), y.to_numpy())
    hgb.fit(X, y)
    return lr, xgb_clf, hgb

def mean_ensemble(*probs):
    return np.column_stack(probs).mean(axis=1)

def stack_probs(base_probs, y, cfg):
    meta = np.column_stack(base_probs)
    stk = LogisticRegression(max_iter=1000, solver="lbfgs",
                             random_state=cfg["seed"]).fit(meta, y)
    return stk.predict_proba(meta)[:, 1]

def fit_ensemble(X, y, cfg):
    lr, xgb_clf, _ = train_base_models(X, y, cfg)
    return 0.5 * (lr.predict_proba(X)[:, 1] +
                  xgb_clf.predict_proba(X.to_numpy())[:, 1])
