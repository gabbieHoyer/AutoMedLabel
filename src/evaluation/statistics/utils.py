
import pandas as pd

def remove_brackets_and_convert(df, subject_column='Subject'):
    """
    Remove brackets from string representations of numbers and convert them to floats.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        subject_column (str): Column name to exclude from cleaning (usually the subject identifier).

    Returns:
        pd.DataFrame: Cleaned DataFrame with numerical values converted to floats.
    """
    # Apply function to each column except the subject column
    for col in df.columns:
        if col != subject_column:
            # Use str to remove brackets and convert to float
            df[col] = df[col].apply(lambda x: float(str(x).strip('[]')) if pd.notnull(x) else x)
    return df
