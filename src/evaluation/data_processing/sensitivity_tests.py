
# src/evaluation/data_processing/sensitivity_tests.py
import pandas as pd
from scipy.stats import ks_2samp

def ks_test_by_group(original_df, imputed_df, group_col, columns_to_test):
    """
    Perform KS tests for each group in group_col and for each specified column.
    Return a DataFrame with the KS statistic and p-value.
    """
    results = []
    for group_name, group in original_df.groupby(group_col):
        imputed_group = imputed_df[imputed_df[group_col] == group_name]
        for column in columns_to_test:
            orig = group[column].dropna()
            imp = imputed_group[column]
            if orig.empty or imp.empty:
                continue
            if not (orig.dtype.kind in 'biufc' and imp.dtype.kind in 'biufc'):
                print(f"Skipping non-numeric column: {column} in group: {group_name}")
                continue
            ks_stat, p_val = ks_2samp(orig, imp)
            results.append({
                'Group': group_name,
                'Column': column,
                'KS Statistic': ks_stat,
                'P-Value': p_val
            })
    return pd.DataFrame(results)

def ks_test_global(original_df, imputed_df, columns_to_test):
    """
    Perform KS tests globally (without grouping).
    """
    results = []
    for column in columns_to_test:
        orig = original_df[column].dropna()
        imp = imputed_df[column]
        if orig.empty or imp.empty:
            continue
        if not (orig.dtype.kind in 'biufc' and imp.dtype.kind in 'biufc'):
            print(f"Skipping non-numeric column: {column}")
            continue
        ks_stat, p_val = ks_2samp(orig, imp)
        results.append({
            'Column': column,
            'KS Statistic': ks_stat,
            'P-Value': p_val
        })
    return pd.DataFrame(results)
