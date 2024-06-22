# If you have common functions or 
# utilities that are used across 
# different modules, you might place 
# them in a utils.py file for shared access.

import re
from collections import defaultdict
import numpy as np
import random


# def extract_numeric_value(value):
#     """Extracts the numeric part of a string and returns it as a float or int, as appropriate."""
#     if isinstance(value, (int, float)):
#         return value  # Return the value directly if it's already a numeric type
#     elif isinstance(value, str):
#         # Extract numeric part, including decimal points
#         numeric_part = re.sub("[^0-9.]", "", value)
#         if numeric_part.isdigit():
#             return int(numeric_part)  # Return as int if there are no decimal points
#         try:
#             return float(numeric_part)  # Attempt to convert to float
#         except ValueError:
#             return None  # Return None if conversion to float fails
#     else:
#         return None  # Return None if value is neither an int, float, nor a str

def extract_numeric_value(value):
    """Extracts the numeric part of a string representing age in format 'XXXY' and returns it as an integer."""
    if isinstance(value, (int, float)):
        return int(value)  # Cast to int if it's already a numeric type
    elif isinstance(value, str):
        # Check if the string is in the expected format with 'Y' at the end
        if value.endswith('Y') and value[:-1].isdigit():
            return int(value[:-1])  # Return as int after stripping the last character
        else:
            # Fallback to regex extraction if the format is unexpected
            numeric_part = re.sub("[^0-9]", "", value)
            if numeric_part.isdigit():
                return int(numeric_part)  # Return as int if the part is numeric
            else:
                return None  # Return None if no numeric part is found
    else:
        return None  # Return None if value is neither an int, float, nor a str


def preprocess_subjects(metadata):
    """
    Preprocess subjects to extract and normalize relevant information.
    Returns a list of subjects with 'sex', 'age', and 'weight' keys.
    """
    processed_subjects = []
    for subject_id, data in metadata.items():
        # sex = data.get('Sex', 'Unknown')  
        sex_raw = data.get('Sex', 'Other')  # Default to 'Other' if 'Sex' is not present
        # Assign 'Other' to any value that is not 'F' or 'M'
        sex = sex_raw if sex_raw in ['F', 'M'] else 'Other'

        age = extract_numeric_value(data.get('Age'))
        weight = extract_numeric_value(data.get('Weight'))
        
        if age is None or weight is None:
            continue  # Skip if age or weight is not provided or not in a recognizable format

        processed_subjects.append({
            'subject_id': subject_id,
            'Sex': sex,
            'Age': age,
            'Weight': weight
        })
    
    return processed_subjects # look clean 

def stratify_and_sample(subjects, split_ratios, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    stratified = {'M': [], 'F': [], 'Other': []}
    for subject in subjects:
        stratified[subject['Sex']].append(subject)

    split_assignments = {'train': [], 'val': [], 'test': []}
    total_subjects = sum(len(group) for group in stratified.values())

    if total_subjects == 0:
        raise ValueError("No subjects to split, please check your subjects list.")

    for sex, group_subjects in stratified.items():
        np.random.shuffle(group_subjects)  # Ensure random distribution
        n = len(group_subjects)
        
        train_count = int(n * split_ratios['train'])
        val_count = round(n * split_ratios['val'])
        test_count = n - train_count - val_count  # Ensure total count matches n

        # Correct any potential off-by-one errors due to rounding
        total_assigned = train_count + val_count + test_count
        if total_assigned < n:
            test_count += n - total_assigned

        # Assign subjects to splits
        split_assignments['train'].extend(group_subjects[:train_count])
        split_assignments['val'].extend(group_subjects[train_count:train_count + val_count])
        split_assignments['test'].extend(group_subjects[train_count + val_count:])

    return split_assignments


def adjust_splits_for_balance(split_assignments, sexes):
    # Placeholder for potential adjustments
    # This could involve checking the distribution of ages and weights within each split
    # and swapping subjects between splits to achieve a better balance.
    pass
