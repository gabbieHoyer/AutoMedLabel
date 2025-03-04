
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pingouin as pg
from pingouin.config import options
from statsmodels.regression.mixed_linear_model import MixedLM

def prepare_icc_data(gt_df, pred_df, label, subject_col='Subject'):
    """
    Prepare data for ICC analysis by combining ground truth and prediction data.
    
    Parameters:
        gt_df (pd.DataFrame): Ground truth dataframe.
        pred_df (pd.DataFrame): Prediction dataframe.
        label (str): The column label to analyze.
        subject_col (str): Column name representing the subject.
    
    Returns:
        pd.DataFrame: Combined dataframe with 'Score' and 'Rater' columns.
    """
    gt_data = gt_df[[subject_col, label]].rename(columns={label: 'Score'})
    gt_data['Rater'] = 'Ground Truth'
    pred_data = pred_df[[subject_col, label]].rename(columns={label: 'Score'})
    pred_data['Rater'] = 'Model Prediction'
    data = pd.concat([gt_data, pred_data], ignore_index=True)
    return data

# ---------------------- STANDARD PARAMETRIC ---------------------- #

def perform_icc_analysis(gt_df, pred_df, labels, subject_col='Subject', save_path='.', file_name='ICC_results.csv'):
    """
    Perform ICC analysis on ground truth and prediction dataframes.
    
    Parameters:
        gt_df (pd.DataFrame): Ground truth dataframe.
        pred_df (pd.DataFrame): Prediction dataframe.
        labels (list): List of label columns to analyze.
        subject_col (str): Name of the subject column.
        save_path (str): Directory to save the CSV file.
        file_name (str): Name of the output CSV file.
    
    Returns:
        pd.DataFrame: Restructured ICC results.
    """
    icc_results = []
    ci_decimals = 4

    pg.set_default_options()
    pg.options["round"] = None
    pg.options["round.column.CI95%"] = 6

    for label in labels:
        data = prepare_icc_data(gt_df, pred_df, label, subject_col)
        icc = pg.intraclass_corr(data=data, targets=subject_col, raters='Rater', ratings='Score')
        print(icc)
        icc['CI95%'] = icc['CI95%'].apply(lambda ci: [round(ci[0], ci_decimals), round(ci[1], ci_decimals)])
        icc_results.append({'Label': label, 'ICC': icc})

    icc_results_df = pd.DataFrame(icc_results)
    restructured_data = []
    for index, row in icc_results_df.iterrows():
        label = row['Label']
        icc_table = row['ICC']
        icc_table['Label'] = label
        restructured_data.append(icc_table)
    final_df = pd.concat(restructured_data, ignore_index=True)
    columns_order = ['Label'] + [col for col in final_df.columns if col != 'Label']
    final_df = final_df[columns_order]
    output_file = os.path.join(save_path, file_name)
    final_df.to_csv(output_file, index=False)
    return final_df

def extract_full_icc_info(icc_df, icc_type='ICC3k'):
    """
    Extract specific ICC information for each label from the ICC DataFrame including confidence intervals.
    
    Parameters:
        icc_df (pd.DataFrame): DataFrame containing ICC results for all labels.
        icc_type (str): Type of ICC to extract.
    
    Returns:
        dict: Dictionary containing detailed ICC information for each label.
    """
    icc_info = {}
    filtered_df = icc_df[icc_df['Type'] == icc_type]
    for index, row in filtered_df.iterrows():
        label = row['Label']
        icc_info[label] = {
            'icc': row['ICC'],
            'ci': row['CI95%'],
            'description': row['Description']
        }
    return icc_info

# ---------------------- NON-PARAMETRIC ---------------------- #

def extract_full_icc_info_nonpar(icc_df):
    """
    Extract specific ICC information for each label from the bootstrapped ICC DataFrame including confidence intervals.
    
    Parameters:
        icc_df (pd.DataFrame): DataFrame containing ICC results for all labels.
    
    Returns:
        dict: Dictionary containing detailed ICC information for each label.
    """
    icc_info = {}
    for index, row in icc_df.iterrows():
        label = row['Label']
        icc_info[label] = {
            'mean_icc': row['Mean_ICC'],
            'median_icc': row['Median_ICC'],
            'ci_lower': row['CI_95%_Lower'],
            'ci_upper': row['CI_95%_Upper'],
            'n_bootstraps': row['n_bootstraps']
        }
    return icc_info

def calculate_icc_mixed_model(data, subject_col, rater_col, score_col):
    """
    Calculate the ICC from a linear mixed-effects model.
    
    Parameters:
        data (pd.DataFrame): Data containing the ratings.
        subject_col (str): Column name representing the subject.
        rater_col (str): Column name representing the rater.
        score_col (str): Column name representing the score.
    
    Returns:
        float: The ICC value computed from the variance components.
    """
    try:
        model = MixedLM.from_formula(f'{score_col} ~ 1', groups=subject_col, data=data)
        result = model.fit()
        var_subject = result.cov_re.iloc[0, 0]
        var_residual = result.scale
        icc = var_subject / (var_subject + var_residual)
        return icc
    except Exception as e:
        print(f"Failed to fit mixed model for {score_col}: {e}")
        return np.nan

def bootstrap_icc_mixed_model(gt_df, pred_df, labels, subject_col='Subject', n_bootstraps=10000, save_path='.', file_name='bootstrap_ICC_results.csv', two_sided=False):
    """
    Perform bootstrapped ICC analysis using a linear mixed-effects model on ground truth and prediction dataframes.
    
    Parameters:
        gt_df (pd.DataFrame): Ground truth dataframe.
        pred_df (pd.DataFrame): Prediction dataframe.
        labels (list): List of label columns to analyze.
        subject_col (str): Name of the subject column.
        n_bootstraps (int): Number of bootstrap resamples.
        save_path (str): Directory to save the results.
        file_name (str): Name of the CSV file to save results.
        two_sided (bool): Whether to perform a two-sided p-value calculation.
    
    Returns:
        pd.DataFrame: Bootstrapped ICC results including mean, confidence intervals, p-value, etc.
    """
    bootstrap_results = []

    for label in labels:
        data = prepare_icc_data(gt_df, pred_df, label, subject_col)
        if data.empty:
            print(f"No valid data for label {label}. Skipping this label.")
            bootstrap_results.append({
                'Label': label,
                'Mean_ICC': np.nan,
                'Median_ICC': np.nan,
                'CI_95%_Lower': np.nan,
                'CI_95%_Upper': np.nan,
                'CI_95%': np.nan,
                'p_value': np.nan,
                'n_bootstraps': 0,
                'n_subjects': 0
            })
            continue

        n_subjects = data[subject_col].nunique()
        icc_bootstrap_values = []

        for _ in tqdm(range(n_bootstraps), desc=f'Bootstrapping ICC for {label}'):
            resampled_subjects = np.random.choice(data[subject_col].unique(), size=n_subjects, replace=True)
            resampled_data = data[data[subject_col].isin(resampled_subjects)].copy()
            icc_value = calculate_icc_mixed_model(resampled_data, subject_col, 'Rater', 'Score')
            if not np.isnan(icc_value):
                icc_bootstrap_values.append(icc_value)

        if len(icc_bootstrap_values) > 0:
            icc_bootstrap_values = np.array(icc_bootstrap_values)
            mean_icc = np.mean(icc_bootstrap_values)
            median_icc = np.median(icc_bootstrap_values)
            ci_lower = np.percentile(icc_bootstrap_values, 2.5)
            ci_upper = np.percentile(icc_bootstrap_values, 97.5)
            ci_formatted = f'[{ci_lower:.3f} {ci_upper:.3f}]'
            proportion_le_zero = np.mean(icc_bootstrap_values <= 0)
            if two_sided:
                proportion_ge_icc = np.mean(icc_bootstrap_values >= mean_icc)
                p_value = 2 * min(proportion_le_zero, proportion_ge_icc)
                p_value = min(p_value, 1.0)
            else:
                p_value = proportion_le_zero
        else:
            print(f"Warning: No valid ICC values for {label}. Skipping this label.")
            mean_icc = np.nan
            median_icc = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            ci_formatted = np.nan
            p_value = np.nan

        bootstrap_results.append({
            'Label': label,
            'Mean_ICC': mean_icc,
            'Median_ICC': median_icc,
            'CI_95%_Lower': ci_lower,
            'CI_95%_Upper': ci_upper,
            'CI_95%': ci_formatted,
            'p_value': f"{p_value:.6f}",
            'n_bootstraps': len(icc_bootstrap_values),
            'n_subjects': n_subjects
        })

    bootstrap_results_df = pd.DataFrame(bootstrap_results)
    output_file = os.path.join(save_path, file_name)
    bootstrap_results_df.to_csv(output_file, index=False)
    return bootstrap_results_df

