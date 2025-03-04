
# src/evaluation/data_processing/feature_selection.py
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature.
    Assumes the dataframe includes a constant column.
    """
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def drop_high_vif_features(vif_df, vif_threshold=10):
    """
    Return a list of features with VIF higher than the specified threshold.
    """
    high_vif = vif_df[vif_df['VIF'] > vif_threshold]['feature'].tolist()
    return high_vif

def create_slimmed_df(df, essential_feature_list):
    """
    Return a dataframe that includes only the specified essential features.
    """
    return df[essential_feature_list].copy()
