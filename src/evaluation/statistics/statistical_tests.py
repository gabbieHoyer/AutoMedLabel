
import os
import pandas as pd
from scipy.stats import shapiro, levene, spearmanr

def _validate_and_align_dataframes(df1, df2, subject_column):
    """
    Set the index to the subject column for both dataframes and verify that their indices match.

    Parameters:
        df1 (pd.DataFrame): First dataframe.
        df2 (pd.DataFrame): Second dataframe.
        subject_column (str): Name of the subject column.

    Returns:
        tuple: Two aligned dataframes.
    """
    df1_aligned = df1.set_index(subject_column)
    df2_aligned = df2.set_index(subject_column)
    if not df1_aligned.index.equals(df2_aligned.index):
        raise ValueError("Subject rows do not match between dataframes.")
    return df1_aligned, df2_aligned

def _align_series(series1, series2):
    """
    Drop NaN values from both series and align them by their indices.

    Parameters:
        series1 (pd.Series): First series.
        series2 (pd.Series): Second series.

    Returns:
        tuple: Two aligned series.
    """
    s1 = series1.dropna()
    s2 = series2.dropna()
    return s1.align(s2, join='inner')

# ---------------------- STANDARD PARAMETRIC ---------------------- #

def compute_levenes_test(df1, df2, columns, subject_column='Subject', save_path='.', file_name='levenes_results.csv'):
    """
    Perform Levene's test for homogeneity of variances on specified columns from two dataframes.

    Parameters:
        df1 (pd.DataFrame): First dataframe (e.g., ground truth data).
        df2 (pd.DataFrame): Second dataframe (e.g., prediction data).
        columns (list): List of columns to perform Levene's test on.
        subject_column (str): Column name for the subjects which should align in both dataframes.
        save_path (str): Directory to save the CSV file.
        file_name (str): Name of the output CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the Levene's test results.
    """
    df1, df2 = _validate_and_align_dataframes(df1, df2, subject_column)
    results = []

    for col in columns:
        x, y = _align_series(df1[col], df2[col])
        stat, p_value = levene(x, y)
        results.append({
            'Class': col,
            'Levene_Statistic': stat,
            'P_Value': p_value
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_path, file_name), index=False)
    return results_df

def compute_shapiro_wilk(df_gt, df_pred, columns, subject_column='Subject', save_path='.', file_name='shapiro_results.csv'):
    """
    Perform the Shapiro-Wilk test for normality on both ground truth and predicted values for each class,
    and save the results as a CSV file.

    Parameters:
        df_gt (pd.DataFrame): DataFrame containing the ground truth data.
        df_pred (pd.DataFrame): DataFrame containing the prediction data.
        columns (list): List of columns to perform the Shapiro-Wilk test on.
        subject_column (str): Column name for the subjects which should align in both dataframes.
        save_path (str): Directory to save the CSV file.
        file_name (str): Name of the output CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the Shapiro-Wilk test results.
    """
    df_gt, df_pred = _validate_and_align_dataframes(df_gt, df_pred, subject_column)
    results = []

    for col in columns:
        x, y = _align_series(df_gt[col], df_pred[col])
        stat_gt, p_value_gt = shapiro(x)
        stat_pred, p_value_pred = shapiro(y)
        results.append({
            'Class': col,
            'Shapiro_Statistic_GT': stat_gt,
            'P_Value_GT': p_value_gt,
            'Shapiro_Statistic_Pred': stat_pred,
            'P_Value_Pred': p_value_pred
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_path, file_name), index=False)
    return results_df

# ---------------------- NON-PARAMETRIC ---------------------- #

def compute_spearman_correlation_subject_level(df_gt, df_pred, columns, subject_column, save_path, file_name):
    """
    Compute Spearman's rank correlation coefficient between ground truth and predictions at the subject level.

    Parameters:
        df_gt (pd.DataFrame): Ground truth data.
        df_pred (pd.DataFrame): Predicted data.
        columns (list): List of measurement columns.
        subject_column (str): Subject column name.
        save_path (str): Directory to save the CSV file.
        file_name (str): Output CSV file name.

    Returns:
        pd.DataFrame: DataFrame with Spearman's rho and p-values for each variable.
    """
    results = []
    for col in columns:
        df_merged = pd.merge(
            df_pred[[subject_column, col]],
            df_gt[[subject_column, col]],
            on=subject_column,
            suffixes=('_pred', '_gt')
        )
        df_merged = df_merged.dropna(subset=[f'{col}_gt', f'{col}_pred'])

        if len(df_merged) < 2:
            print(f"Not enough data for Spearman's correlation on '{col}' (n={len(df_merged)}).")
            results.append({
                'Variable': col,
                "Spearman's rho": None,
                'p-value': None
            })
            continue

        rho, p_value = spearmanr(df_merged[f'{col}_gt'], df_merged[f'{col}_pred'])
        results.append({
            'Variable': col,
            "Spearman's rho": rho,
            'p-value': p_value
        })

    results_df = pd.DataFrame(results)
    os.makedirs(save_path, exist_ok=True)
    results_df.to_csv(os.path.join(save_path, file_name), index=False)
    return results_df
