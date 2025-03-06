# src/evaluation/mri_acquisition_analysis.py
import os
import pandas as pd

from src.evaluation.data_processing import (
    load_base_data, select_columns, replace_bbox_shift, drop_duplicates,
    encode_impute_scale, calculate_vif, drop_high_vif_features, create_slimmed_df,
    ks_test_by_group, ks_test_global, encode_and_impute
)
from src.evaluation.mixed_modeling import (
    load_processed_data,  aggregate_data,
    calculate_vif_df, run_mixed_effects_model, plot_fixed_effects, save_model_summary
)
# Import the config dictionary
from src.evaluation.config import config

def data_processing_step(config):
    # Load the base data and select columns.
    df_base = load_base_data(config['base_filepath'])
    df_slim = select_columns(df_base, config['selected_columns'])
    df_slim = replace_bbox_shift(df_slim)
    df_slim = drop_duplicates(df_slim)
    
    # Create the slimmed dataframe using essential features.
    df_post_feat_sel = create_slimmed_df(df_slim, config['essential_features'])
    df_post_feat_sel['bbox_shift'].replace(5, 20, inplace=True)
    
    # First round: perform encoding and imputation (without scaling) using encode_and_impute.
    df_processed = encode_and_impute(
        df_post_feat_sel,
        config['group_col'],  # if needed, or simply pass None if not used.
        config['categorical_nominal_columns'],
        config['categorical_ordinal_columns'],
        config['continuous_numerical_columns'],
        config['discrete_numerical_columns'],
        config['id_columns'],
        config['target_columns']
    )
    
    # (Optional) You can run KS tests on this intermediate version if desired.
    # Then, prepare for the second round by dropping problematic columns.
    df_post_feat_sel_clean = df_post_feat_sel.drop(columns=config['columns_to_drop'], errors='ignore')
    
    # Create cleaned lists by excluding the problematic columns.
    continuous_clean = [col for col in config['continuous_numerical_columns'] if col not in config['columns_to_drop']]
    discrete_clean = [col for col in config['discrete_numerical_columns'] if col not in config['columns_to_drop']]
    
    # Second round: perform encoding, imputation, and scaling on the cleaned dataframe.
    df_processed_clean = encode_impute_scale(
        df_post_feat_sel_clean,
        config['categorical_nominal_columns'],
        config['categorical_ordinal_columns'],
        continuous_clean,
        discrete_clean,
        config['id_columns'],
        config['target_columns']
    )
    
    # Optionally run KS tests comparing before and after cleaning.
    columns_to_test = continuous_clean + discrete_clean
    ks_results_group = ks_test_by_group(df_post_feat_sel_clean, df_processed_clean, config['ks_group_col'], columns_to_test)
    print("KS Test by group results:\n", ks_results_group)
    ks_results_global = ks_test_global(df_post_feat_sel_clean, df_processed_clean, columns_to_test)
    print("Global KS Test results:\n", ks_results_global)
    
    # Save the final processed (clean) dataframe.
    os.makedirs(config['output_dir'], exist_ok=True)
    processed_filepath = os.path.join(config['output_dir'], 'processed_data_new.csv')
    df_processed_clean.to_csv(processed_filepath, index=False)
    
    return df_processed_clean

def mixed_modeling_step(df_processed, config):
    """
    This function runs the mixed effects modeling step.
    It aggregates the processed dataframe based on user-supplied grouping columns,
    then loops over each non-baseline experiment type (as specified in the config) to fit a model,
    save its summary, and plot fixed effects.
    """
    # Aggregate data to dataset level using the grouping columns from the config.
    df_dataset_level = aggregate_data(df_processed, config['group_columns'])
    print("Dataset-level aggregated data:\n", df_dataset_level.head())
    
    os.makedirs(config['mixed_output_dir'], exist_ok=True)
    
    # Identify experiment types that are not the baseline.
    experiment_types = df_dataset_level[config['experiment_type_col']].unique()
    experiment_types = [et for et in experiment_types if et != config['baseline_value']]
    print(f"Experiment types (non-baseline): {experiment_types}")
    
    results = {}
    for et in experiment_types:
        df_subgroup = df_dataset_level[df_dataset_level[config['experiment_type_col']] == et]
        df_baseline = df_dataset_level[df_dataset_level[config['experiment_type_col']] == config['baseline_value']]
        df_combined = pd.concat([df_subgroup, df_baseline])
        
        result = run_mixed_effects_model(df_combined, config['model_formula'], config['group_col'], output_dir=config['mixed_output_dir'])
        results[et] = result
        save_model_summary(result, et, output_dir=config['mixed_output_dir'])
        print(f"Model results for experiment type {et} vs. baseline:")
        print(result.summary())
    
    for et, res in results.items():
        plot_fixed_effects(res, et, output_dir=config['mixed_output_dir'])
    
    return results

def bonus_experiment_filter(df, output_path):
    filtered_data2 = df[
    (df['balanced_True'] == 0) &
    ((df['experiment_type_new'] == 0) | (df['max_subject_set'] == 5))
]
    filtered_data2.rename(columns={'2D_3D_3D':'twoD_threeD_threeD'}, inplace=True)
    filtered_data2['image_encoder_True'] = filtered_data2['image_encoder_True'].astype(int)
    filtered_data2['balanced_True'] = filtered_data2['balanced_True'].astype(int)

    # Ensure experiment_type_id is numeric
    filtered_data2['experiment_type_id_new'] = pd.to_numeric(filtered_data2['experiment_type_id_new'], errors='coerce')
    filtered_data2 = filtered_data2.dropna()  
    
    filtered_data2.to_csv(output_path, index=False)
    return filtered_data2

def main():
    run_full_pipeline = True  # Set to False to skip data processing if processed data already exists.
    
    if run_full_pipeline:
        df_processed = data_processing_step(config)
    else:
        processed_filepath = os.path.join(config['output_dir'], 'processed_data_new.csv')
        df_processed = pd.read_csv(processed_filepath)
    
    exp_processed_filepath = os.path.join(config['output_dir'], 'experiment_processed_data.csv')
    df_processed = bonus_experiment_filter(df_processed, exp_processed_filepath)

    mixed_modeling_step(df_processed, config)

if __name__ == '__main__':
    main()

