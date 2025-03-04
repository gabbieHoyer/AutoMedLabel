
import os
import pandas as pd
from scipy import stats

# ---------------------- STANDARD PARAMETRIC ---------------------- #

def compute_regression_results(df_gt, df_pred, columns, subject_column='Subject', save_path='.', file_name='regression_results.csv'):
    """
    Compute regression statistics for each class and save the results as a CSV file.

    Parameters:
        df_gt (pd.DataFrame): DataFrame containing the ground truth data.
        df_pred (pd.DataFrame): DataFrame containing the prediction data.
        columns (list): List of columns (classes) to perform regression analysis on.
        subject_column (str): Column name for the subjects which should align in both dataframes.
        save_path (str): Path to save the CSV file.
        file_name (str): Name of the output CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the regression results.
    """
    # Ensure the subjects are aligned
    df_gt = df_gt.set_index(subject_column)
    df_pred = df_pred.set_index(subject_column)

    # Verify that the dataframes align properly
    assert (df_gt.index == df_pred.index).all(), "Subject rows do not match between dataframes."

    # Initialize list to store the results
    results = []

    # Compute regression statistics for each class
    for col in columns:
        x = df_gt[col].dropna()
        y = df_pred[col].dropna()

        x, y = x.align(y, join='inner')

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate R²
        r_squared = r_value**2

        # Append the results to the list
        results.append({
            'Class': col,
            'Intercept': intercept,
            'Slope': slope,
            'R_value': r_value,
            'R_squared': r_squared,
            'P_value': p_value,
            'Std_err': std_err
        })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    # # Save the results to a CSV file
    output_file = os.path.join(save_path, file_name)
    results_df.to_csv(output_file, index=False)

    # Return the results DataFrame
    return results_df

# ---------------------- NON-PARAMETRIC ---------------------- #

