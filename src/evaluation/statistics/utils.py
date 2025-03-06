
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


try:
    from rich.console import Console
    from rich.table import Table
    from rich.theme import Theme

    RICH_AVAILABLE = True
    
    # Customize this theme as desired
    custom_theme = Theme({"header": "bold green"})
    console = Console(theme=custom_theme)

except ImportError:
    RICH_AVAILABLE = False
    console = None


def console_rule(title: str, style="header"):
    """
    Print a rule (horizontal line) with a title at the center.
    If Rich is available, use console.rule with the given style.
    Otherwise, fall back to a simple line of text.
    """
    if RICH_AVAILABLE:
        console.rule(f"[{style}]{title}")
    else:
        print("\n" + "="*10 + f" {title} " + "="*10 + "\n")


def console_print(message: str):
    """
    Print a message using Rich if available.
    Otherwise, use a normal print.
    """
    if RICH_AVAILABLE:
        console.print(message)
    else:
        print(message)


def print_df_as_table(df, title="DataFrame", max_width=20):
    """
    Print a pandas DataFrame in a Rich table if available,
    otherwise print a plain text DataFrame.
    """
    if RICH_AVAILABLE:
        table = Table(title=title, show_lines=True)
        
        # Add columns with overflow and max width
        for col in df.columns:
            table.add_column(str(col), overflow="fold", max_width=max_width)
        
        # Add rows
        for _, row in df.iterrows():
            row_strs = [str(item) for item in row]
            table.add_row(*row_strs)
        
        console.print(table)
    else:
        print(f"\n{title}\n{df.to_string()}\n")


def icc_dict_to_dataframe(icc_dict):
    """
    Convert an ICC dictionary like:
        {
            'femoral cartilage': {
                'icc': 0.9889991508767437,
                'ci': [0.9661, 0.9965],
                'description': 'Single fixed raters'
            },
            ...
        }
    to a pandas DataFrame with columns:
        [label, icc, ci_lower, ci_upper, description].
    """
    df = pd.DataFrame.from_dict(icc_dict, orient='index')  # each key becomes an index row
    df.index.name = 'label'
    df.reset_index(inplace=True)
    
    # Expand the CI list into two separate columns
    df['ci_lower'] = df['ci'].apply(lambda x: x[0])
    df['ci_upper'] = df['ci'].apply(lambda x: x[1])
    
    # Drop the original 'ci' column
    df.drop(columns=['ci'], inplace=True)
    
    # Reorder columns to something more logical
    df = df[['label', 'icc', 'ci_lower', 'ci_upper', 'description']]
    return df
