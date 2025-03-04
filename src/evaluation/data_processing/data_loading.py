
# src/evaluation/data_processing/data_loading.py
import pandas as pd

def load_base_data(filepath):
    """
    Read the base CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def select_columns(df, columns_list):
    """
    Return a slimmed-down dataframe with only the specified columns.
    """
    return df[columns_list].copy()

def replace_bbox_shift(df, column='bbox_shift', replace_from=5, replace_to=20):
    """
    Replace specific values in the bbox_shift column.
    """
    df[column].replace(replace_from, replace_to, inplace=True)
    return df

def drop_duplicates(df):
    """
    Drop duplicate rows from the dataframe.
    """
    return df.drop_duplicates()
