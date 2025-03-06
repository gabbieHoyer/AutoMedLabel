# src/evaluation/biomarker_analysis.py

import os
import pandas as pd
from os.path import join

from src.evaluation.statistics import (
    remove_brackets_and_convert, icc_dict_to_dataframe,
    compute_shapiro_wilk, compute_levenes_test, compute_spearman_correlation_subject_level,
    perform_icc_analysis, extract_full_icc_info,
    bootstrap_icc_mixed_model, extract_full_icc_info_nonpar,
    compute_regression_results,
    plot_bland_altman_multiple, plot_regression_comparison,
    plot_gp_regression_subject_level, plot_bland_altman_multiple_nonparametric_subject_level,
    console_rule, console_print, print_df_as_table
)

from src.utils import get_project_root
root = get_project_root()

class AnalysisPipeline:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.biomarker = config['biomarker']
        self.subject_column = config['subject_column']
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()

    def load_data(self):
        self.df_gt_slice = pd.read_csv(self.config['gt_slice_path'])
        self.df_gt_subject = pd.read_csv(self.config['gt_subject_path'])
        self.df_pred_slice = pd.read_csv(self.config['pred_slice_path'])
        self.df_pred_subject = pd.read_csv(self.config['pred_subject_path'])
        if self.config.get('clean_data', False):
            print("Cleaning data by removing brackets and converting to floats...")
            self.df_gt_slice = remove_brackets_and_convert(self.df_gt_slice, subject_column=self.subject_column)
            self.df_gt_subject = remove_brackets_and_convert(self.df_gt_subject, subject_column=self.subject_column)
            self.df_pred_slice = remove_brackets_and_convert(self.df_pred_slice, subject_column=self.subject_column)
            self.df_pred_subject = remove_brackets_and_convert(self.df_pred_subject, subject_column=self.subject_column)
            print("Data cleaning completed.")
        # Use subject-level data for analysis
        self.df_ground_truth = self.df_gt_subject.copy()
        self.df_prediction = self.df_pred_subject.copy()

    def run(self):
        raise NotImplementedError("Subclasses must implement the run method.")

class ParametricAnalysis(AnalysisPipeline):
    def run(self):
        # Statistical Tests
        console_rule("Shapiro-Wilk Test:")
        shapiro_results_df = compute_shapiro_wilk(
            self.df_ground_truth,
            self.df_prediction,
            self.config['columns'],
            subject_column=self.subject_column,
            save_path=self.output_dir,
            file_name='shapiro_results.csv'
        )
        print_df_as_table(shapiro_results_df, title="Shapiro-Wilk Results", max_width=20)

        console_rule("Levene Test:")
        levene_results_df = compute_levenes_test(
            self.df_ground_truth,
            self.df_prediction,
            self.config['columns'],
            subject_column=self.subject_column,
            save_path=self.output_dir,
            file_name='levene_results.csv'
        )
        print_df_as_table(levene_results_df, title="Levene Test Results", max_width=20)

        # ICC Analysis
        console_rule("ICC Analysis:")
        icc_results_df = perform_icc_analysis(
            self.df_ground_truth,
            self.df_prediction,
            self.config['columns'],
            subject_col=self.subject_column,
            save_path=self.output_dir,
            file_name='icc_results.csv'
        )
        icc_values = extract_full_icc_info(icc_results_df, icc_type='ICC3')
        print_df_as_table(icc_dict_to_dataframe(icc_values), title="ICC Results", max_width=20)

        # Regression Analysis
        console_rule("Regression Analysis:")
        regression_results_df = compute_regression_results(
            df_gt=self.df_ground_truth,
            df_pred=self.df_prediction,
            columns=self.config['columns'],
            subject_column=self.subject_column,
            save_path=self.output_dir,
            file_name='regression_results.csv'
        )
        print_df_as_table(regression_results_df, title="Regression Results", max_width=20)

        # Plotting
        console_rule("Generate Bland-Altman Plots:")
        plot_bland_altman_multiple(
            df_pred=self.df_prediction,
            df_gt=self.df_ground_truth,
            columns=self.config['columns'],
            subject_column=self.subject_column,
            save_path=os.path.join(self.output_dir, 'bland_altman_plots'),
            dpi=300,
            dataset_name=self.dataset_name,
            biomarker=self.biomarker,
            units=self.config.get('units', None)
        )
        print("Bland-Altman Complete.")

        console_rule("Generate Regression Plots:")
        plot_regression_comparison(
            df_gt=self.df_ground_truth,
            df_pred=self.df_prediction,
            columns=self.config['columns'],
            icc_info=icc_values,
            regression_results_df=regression_results_df,
            subject_column=self.subject_column,
            save_path=os.path.join(self.output_dir, 'regression_plots'),
            dpi=300,
            dataset_name=self.dataset_name,
            biomarker=self.biomarker,
            units=self.config.get('units', None)
        )
        print("Regression Complete.")

class NonParametricAnalysis(AnalysisPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.n_bootstraps = config['n_bootstraps']  # or use config.get('n_bootstraps', default_value)

    def run(self):
        # Statistical Tests: 
        console_rule("Spearman's Rank Correlation:")
        spearman_results_df = compute_spearman_correlation_subject_level(
            self.df_ground_truth,
            self.df_prediction,
            self.config['columns'],
            subject_column=self.subject_column,
            save_path=self.output_dir,
            file_name='spearman_results.csv'
        )
        print_df_as_table(spearman_results_df, title="Spearman's Rank Correlation", max_width=20)

        # ICC Analysis: Bootstrap approach
        console_rule("Bootstrap ICC Analysis:")
        icc_results_df = bootstrap_icc_mixed_model(
            self.df_ground_truth,
            self.df_prediction,
            self.config['columns'],
            subject_col=self.subject_column,
            n_bootstraps=self.n_bootstraps,
            save_path=self.output_dir,
            file_name='icc_results_median.csv'
        )
        icc_values = extract_full_icc_info_nonpar(icc_results_df)
        print_df_as_table(icc_results_df, title="Bootstrap ICC Results", max_width=20)

        # Plotting:
        console_rule("Non-Parametric Bland-Altman Analysis:")
        plot_bland_altman_multiple_nonparametric_subject_level(
            df_pred=self.df_prediction,
            df_gt=self.df_ground_truth,
            columns=self.config['columns'],
            subject_column=self.subject_column,
            n_bootstraps=self.n_bootstraps,
            save_path=os.path.join(self.output_dir, f'bland_altman_nonparametric_{self.n_bootstraps}'),
            dataset_name=self.dataset_name,
            biomarker=self.biomarker,
            units=self.config.get('units', None)
        )
        print("Non-parametric Bland-Altman plots generated.")

        # Plotting: Gaussian Process Regression Comparison
        console_rule("Gaussian Process Regression Analysis:")
        gp_save_path = os.path.join(self.output_dir, 'regression_comparisons_gp_median')
        plot_gp_regression_subject_level(
            df_gt=self.df_ground_truth,
            df_pred=self.df_prediction,
            columns=self.config['columns'],
            spearman_results_df=spearman_results_df,
            icc_info=icc_values,
            subject_column=self.subject_column,
            save_path=gp_save_path,
            dpi=300,
            dot_color=self.config.get('dot_color', '#4BB98A'),
            line_color='#ff7f0e',
            units=self.config.get('units', None),
            dataset_name=self.dataset_name,
            biomarker=self.biomarker
        )
        print("GP regression comparison plots generated.")

def run_analysis(config, analysis_type='parametric'):
    if analysis_type == 'parametric':
        analysis = ParametricAnalysis(config)
    else:
        analysis = NonParametricAnalysis(config)
    analysis.run()

if __name__ == '__main__':
    # Separate configuration lists can be used for each analysis type
    parametric_configs = [
        {
            'dataset_name': 'Knee_3D_DESS_Research_Anatomical_86',
            'biomarker': r'$\bar{x}$ Cartilage Thickness',
            'gt_slice_path': join(root, 'demos', 'demo_data', 'OAI_Knee_cart_thickness', 'cartilagethicknessmetric_gt_slice_volume.csv'),
            'gt_subject_path': join(root, 'demos', 'demo_data', 'OAI_Knee_cart_thickness', 'cartilagethicknessmetric_gt_volume.csv'),
            'pred_slice_path': join(root, 'demos', 'demo_data', 'OAI_Knee_cart_thickness', 'cartilagethicknessmetric_pred_slice_volume.csv'),
            'pred_subject_path': join(root, 'demos', 'demo_data', 'OAI_Knee_cart_thickness', 'cartilagethicknessmetric_pred_volume.csv'),
            'columns': ['femoral cartilage', 'lateral tibial cartilage', 'medial tibial cartilage', 'patellar cartilage'],
            'subject_column': 'Subject',
            'output_dir': join(root, 'work_dir', 'evaluation', 'stats', 'OAI_Knee_cart_thickness', 'subject'),
            'dot_color': '#4BB98A',
            'move_text': True,
            'clean_data': True,
            'units': 'mm'
        },
        # ... other parametric configurations ...
    ]

    nonparametric_configs = [
        {
            'dataset_name': 'Knee_2D_MAPSS-echo1_Research_Compositional_39',
            'biomarker': r'$\bar{x}$ $T_1\rho$',
            'gt_slice_path': join(root, 'demos', 'demo_data', 'Knee_T1rho_T2_Mapss', 't1rhometric_gt_slice_volume.csv'),
            'gt_subject_path': join(root, 'demos', 'demo_data', 'Knee_T1rho_T2_Mapss', 't1rhometric_gt_volume.csv'),
            'pred_slice_path': join(root, 'demos', 'demo_data', 'Knee_T1rho_T2_Mapss', 't1rhometric_pred_slice_volume.csv'),
            'pred_subject_path': join(root, 'demos', 'demo_data', 'Knee_T1rho_T2_Mapss', 't1rhometric_pred_volume.csv'),
            'columns': ['medial femoral', 'lateral femoral', 'lateral tibial', 'medial tibial', 'trochlear', 'patellar'],
            'subject_column': 'Subject',
            'n_bootstraps':1000,
            'output_dir': join(root, 'work_dir', 'evaluation', 'stats', 'Knee_T1rho_T2_Mapss', 'T1Rho', 'subject_non_parametric'),
            'dot_color': '#a347d1',
            'move_text': False,
            'clean_data': True,
            'units': 'ms'
        },
        # ... other nonparametric configurations ...
    ]

    # Run both pipelines
    for config in parametric_configs:
        print(f"Processing {config['dataset_name']} parametric...")
        run_analysis(config, analysis_type='parametric')
        print(f"{config['dataset_name']} parametric processing completed.\n")
    
    for config in nonparametric_configs:
        print(f"Processing {config['dataset_name']} nonparametric...")
        run_analysis(config, analysis_type='nonparametric')
        print(f"{config['dataset_name']} nonparametric processing completed.\n")


# in root, run python -m src.evaluation.biomarker_analysis
