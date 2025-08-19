import ast, numpy as np, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def to_list(x):
    return x if isinstance(x, list) else ast.literal_eval(x)

def prepare_dataframe(df_raw: pd.DataFrame, cfg) -> pd.DataFrame:
    # ----- field constants pulled from notebook -----
    BONE = ['tissuevolumemetricinference_femur',
            'tissuevolumemetricinference_tibia',
            'tissuevolumemetricinference_patella']
    CART = ['cartilagethicknessmetricinference_femoral cartilage',
            'cartilagethicknessmetricinference_tibial cartilage',
            'cartilagethicknessmetricinference_patellar cartilage']
    METRIC_COLS = BONE + CART

    ANOM_MAP = {
        'femur_anom':  ['MedFemBone','LatFemBone','TroFemBone',
                        'MedFemCart','LatFemCart','TroFemCart'],
        'tibia_anom':  ['MedTibBone','LatTibBone','MedTibCart','LatTibCart'],
        'patella_anom':['PatellaBone','PatellaCart']
    }
    ALL_CLASSES = sorted({lbl for v in ANOM_MAP.values() for lbl in v})

    df = df_raw.copy()

    # ‑‑ multilabel one‑hot
    df["multilabel_list"] = df["multilabel"].apply(to_list)
    mlb = MultiLabelBinarizer(classes=ALL_CLASSES)
    hot = pd.DataFrame(mlb.fit_transform(df["multilabel_list"]),
                       columns=mlb.classes_, index=df.index)
    for col, lbls in ANOM_MAP.items():
        df[col] = hot[lbls].any(axis=1).astype(int)
    df.drop(columns="multilabel_list", inplace=True)

    # ‑‑ z‑scores
    healthy = df[(df["bone_label"] == 0) & (df["cart_label"] == 0)]
    means, sds = healthy[METRIC_COLS].mean(), healthy[METRIC_COLS].std()
    for col in METRIC_COLS:
        df[f"{col}_z"] = (df[col] - means[col]) / sds[col]

    # sex -> numeric
    df["sex_num"] = df["sex"].replace({"M": 0, "F": 1, "0": np.nan}).astype(float)
    df = df.dropna(subset=["sex_num"])

    return df
