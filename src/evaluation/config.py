# src/evaluation/config.py

config = {
    # File paths
    'base_filepath': '/data/mskprojects/mskSAM/users/ghoyer/backup_repo/AutoMedLabel/work_dir/evaluation/mixed_modeling/mskSAM_subject_subcat_expGlobal_dice_cleaned_withBase_sam2_robustID_map2.csv',
    
    # Column groups for data selection
    'experiment_columns': [
        'subject_id', 'eval_run', 'balanced', 'bbox_shift', 'dataset', 'image_encoder', 'mask_decoder',
        'max_subject_set', 'model_ID', 'experiment_name', 'experiment_type', 'max_subject_set_id',
        'experiment_type_id', 'experiment_type_id_new', 'dataset_id', 'Anatomy_id', 'experiment_type_new'
    ],
    'demographic_feats': ['Sex', 'Weight', 'Age', 'Sex_num'],
    'mri_feats': [
        '2D_3D', 'TE', 'TR', 'flip_angle', 'ETL', 'field_strength', 'receive_coil', 'scanner_name', 'scanner_model', 'scanner_vendor',
        'slice_thickness', 'slice_spacing', 'rows', 'columns', 'SAR', 'percent_phase_FOV', 'mri_sequence', 'pixel_spacing_x'
    ],
    'dataset_wholistic_feats': ['Anatomy', 'total_training_subject_volume', 'total_training_slice_number', 'Segmentation_Labels', 'Num_Labels'],
    
    # Combined columns for data selection and for the slimmed dataframe.
    'selected_columns': None,      # Will be computed below.
    'essential_features': None,    # Define essential features as needed.
    
    # Column definitions for preprocessing
    'categorical_nominal_columns': ['balanced', 'bbox_shift', 'image_encoder', '2D_3D', 'dataset', 'Anatomy', 'scanner_vendor'],
    'categorical_ordinal_columns': ['max_subject_set', 'experiment_type_new'],
    'continuous_numerical_columns': ['Weight', 'TE', 'SAR', 'pixel_spacing_x', 'slice_thickness'],
    'discrete_numerical_columns': ['Sex_num', 'Age', 'flip_angle', 'ETL', 'rows', 'Num_Labels'],
    'id_columns': ['subject_id', 'eval_run', 'model_ID', 'dataset_id', 'Anatomy_id', 'experiment_name',
                   'experiment_type_id_new', 'max_subject_set_id', 'total_training_subject_volume', 'total_training_slice_number'],
    'target_columns': ['subject_mean_dice', 'dataset_exp_global_dice'],
    
    # Other configuration values
    'output_dir': '/data/mskprojects/mskSAM/users/ghoyer/backup_repo/AutoMedLabel/work_dir/evaluation/mixed_modeling/',
    'columns_to_drop': ['SAR', 'ETL'],  # Problematic columns to drop in the second round.
    'ks_group_col': 'dataset_id',
    
    # Mixed modeling configuration
    'group_columns': [
        'dataset_id', 'experiment_name', 'eval_run', 'model_ID', 'experiment_type_id_new', 'max_subject_set_id',
        'balanced_True', 'bbox_shift_20', 'image_encoder_True', 'max_subject_set', 'experiment_type_new',
        'total_training_subject_volume', 'total_training_slice_number', 'Num_Labels'
    ],
    'experiment_type_col': 'experiment_type_new',
    'baseline_value': 0,
    'model_formula': """
        dataset_exp_global_dice ~ TE + pixel_spacing_x + slice_thickness + flip_angle + twoD_threeD_threeD +
                                 TE:pixel_spacing_x + TE:slice_thickness +
                                 TE:experiment_type_new + pixel_spacing_x:experiment_type_new +
                                 flip_angle:experiment_type_new +
                                 twoD_threeD_threeD:experiment_type_new +
                                 TE:image_encoder_True + TE:bbox_shift_20 +
                                 flip_angle:image_encoder_True + flip_angle:bbox_shift_20 +
                                 twoD_threeD_threeD:image_encoder_True + twoD_threeD_threeD:bbox_shift_20 +
                                 C(experiment_type_new, Treatment(reference=0))
    """,
    'group_col': 'dataset_id',
    'mixed_output_dir': '/data/mskprojects/mskSAM/users/ghoyer/backup_repo/AutoMedLabel/work_dir/evaluation/mixed_modeling/stats_model_results'
}

# Build combined lists:
config['selected_columns'] = (
    config['experiment_columns'] +
    config['demographic_feats'] +
    config['dataset_wholistic_feats'] +
    config['mri_feats'] +
    ['subject_mean_dice', 'dataset_exp_global_dice']
)

# Define essential features (this may be a subset of selected_columns).
config['essential_features'] = [
    'subject_id', 'eval_run', 'balanced', 'bbox_shift', 'dataset', 'image_encoder', 'mask_decoder', 'max_subject_set',
    'model_ID', 'experiment_name', 'max_subject_set_id', 'experiment_type_id_new', 'dataset_id', 'Anatomy_id', 'experiment_type_new',
    'Weight', 'Age', 'Sex_num', 'Anatomy', 'total_training_subject_volume', 'total_training_slice_number', 'Num_Labels',
    'TE', 'TR', 'flip_angle', 'ETL', 'field_strength', 'receive_coil', 'scanner_name', 'scanner_model', 'scanner_vendor',
    'slice_thickness', 'slice_spacing', 'rows', 'columns', 'SAR', 'percent_phase_FOV', 'mri_sequence', 'pixel_spacing_x',
    '2D_3D', 'subject_mean_dice', 'dataset_exp_global_dice'
]
