
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import transforms

import pingouin as pg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ---------------------- HELPER FUNCTIONS ---------------------- #

def _sanitize_filename(name):
    """Return a sanitized filename by replacing illegal characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def _initialize_plot(save_path, figsize=(8, 6)):
    """
    Ensure the save directory exists, set a seaborn style, and create a figure and axis.
    """
    os.makedirs(save_path, exist_ok=True)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def _finalize_and_save_plot(fig, base_filename, save_path, dpi, show=False):
    """
    Apply common layout adjustments and save the current figure as PNG and SVG.
    Optionally, display the plot.
    """
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    sanitized_name = _sanitize_filename(base_filename)
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(save_path, f"{sanitized_name}.{ext}"), format=ext, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

# ---------------------- STANDARD PARAMETRIC ---------------------- #

def plot_bland_altman_multiple(df_pred, df_gt, columns, subject_column='Subject', save_path='.', dpi=300, dot_color='#a347d1', move_text=False, units=None, dataset_name=None, biomarker=None):
    df_gt = df_gt.set_index(subject_column)
    df_pred = df_pred.set_index(subject_column)
    assert (df_gt.index == df_pred.index).all(), "Subject rows do not match between dataframes."

    for col in columns:
        # Prepare and align data
        x = df_pred[col].dropna()
        y = df_gt[col].dropna()
        x, y = x.align(y, join='inner')

        # Create figure and axis using our helper
        fig, ax = _initialize_plot(save_path, figsize=(8, 6))
        
        # Plot Bland-Altman using Pingouin
        ax = pg.plot_blandaltman(x, y, ax=ax, color='gray', confidence=0.95)
        ax.collections[0].set_color(dot_color)
        ax.collections[0].set_edgecolor('#000000')
        ax.collections[0].set_alpha(0.95)
        ax.collections[0].set_sizes([40])
        
        if len(ax.patches) >= 3:
            ax.patches[0].set_facecolor('tab:grey')
            ax.patches[1].set_facecolor(dot_color)
            ax.patches[2].set_facecolor(dot_color)
        
        if move_text:
            for text in ax.texts:
                if text.get_ha() == 'right':
                    xloc_new = text.get_position()[0] + 0.15
                    text.set_x(xloc_new)
        
        # Set title and labels
        plot_title = f'{dataset_name}:\n{col.capitalize()} ({biomarker})'
        ax.set_title(plot_title, fontsize=16)
        if units:
            x_label = f'Mean of Automated method and Manual annotation ({units})'
            y_label = f'Automated method - Manual annotation ({units})'
        else:
            x_label = 'Mean of Automated method and Manual annotation'
            y_label = 'Automated method - Manual annotation'
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        
        # Save (and show) the plot using our helper
        _finalize_and_save_plot(fig, f'{col}_bland_altman', save_path, dpi, show=True)


def plot_regression_comparison(df_gt, df_pred, columns, icc_info, regression_results_df, subject_column='Subject', save_path='.', dpi=300, dot_color='#a347d1', line_color='#000000', units=None, dataset_name=None, biomarker=None):
    df_gt = df_gt.set_index(subject_column)
    df_pred = df_pred.set_index(subject_column)
    assert (df_gt.index == df_pred.index).all(), "Subject rows do not match between dataframes."

    for col in columns:
        # Prepare and align data
        x = df_pred[col].dropna()
        y = df_gt[col].dropna()
        x, y = x.align(y, join='inner')

        # Retrieve regression parameters
        regression_row = regression_results_df[regression_results_df['Class'] == col].iloc[0]
        intercept = regression_row['Intercept']
        slope = regression_row['Slope']
        r_value = regression_row['R_value']
        r_squared = regression_row['R_squared']
        p_value = regression_row['P_value']

        # Create figure and axis
        fig, ax = _initialize_plot(save_path, figsize=(8, 6))
        
        sns.scatterplot(x=x, y=y, ax=ax, color=dot_color, edgecolor='#000000', alpha=0.9, s=40, linewidth=0.5)
        # Plot identity line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1, label='Identity Line (y = x)')
        
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, '--', color=line_color, linewidth=2, label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
        
        # Retrieve ICC information for annotation
        icc_value = icc_info[col]['icc']
        ci = icc_info[col]['ci']
        p_value_text = 'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
        
        # Position annotation
        x_text_position = np.min(x_vals) + (np.max(x_vals) - np.min(x_vals)) * 0.05
        y_text_position = np.max(y_vals) - (np.max(y_vals) - np.min(y_vals)) * 0.1
        ax.text(x_text_position, y_text_position,
                f'$R^2 = {r_squared:.3f}$, {p_value_text}\n\nICC = {icc_value:.3f} (95% CI: {ci})',
                fontsize=12, va='top', ha='left')
        
        # Set title and labels
        plot_title = f'{dataset_name}:\n{col.capitalize()} ({biomarker})'
        ax.set_title(plot_title, fontsize=16)
        if units:
            x_label = f'Automated method ({units})'
            y_label = f'Manual annotation ({units})'
        else:
            x_label = 'Automated method'
            y_label = 'Manual annotation'
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.legend(fontsize=12, loc='lower right')
        
        _finalize_and_save_plot(fig, f'{col}_regression_comparison', save_path, dpi, show=True)


# ---------------------- NON-PARAMETRIC ---------------------- #

def plot_gp_regression_subject_level(df_gt, df_pred, columns, spearman_results_df, icc_info=None, subject_column='SubjectID', save_path='.', dpi=300, dot_color='#a347d1', line_color='#ff7f0e', units=None, dataset_name=None, biomarker=None):
    os.makedirs(save_path, exist_ok=True)

    for col in columns:
        df_merged = pd.merge(
            df_pred[[subject_column, col]],
            df_gt[[subject_column, col]],
            on=subject_column,
            suffixes=('_pred', '_gt')
        )
        df_merged = df_merged.dropna(subset=[f'{col}_gt', f'{col}_pred'])
        if df_merged.empty:
            print(f"No data available for column '{col}'. Skipping plot.")
            continue

        X = df_merged[f'{col}_pred'].values.reshape(-1, 1)
        y = df_merged[f'{col}_gt'].values

        # Retrieve Spearman's correlation information
        spearman_row = spearman_results_df[spearman_results_df['Variable'] == col]
        if not spearman_row.empty:
            spearman_rho = spearman_row["Spearman's rho"].values[0]
            spearman_p = spearman_row['p-value'].values[0]
            spearman_p_text = 'p < 0.001' if spearman_p < 0.001 else f'p = {spearman_p:.3f}'
        else:
            spearman_rho, spearman_p_text = None, 'p = N/A'

        # Define and fit Gaussian Process
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
        gp.fit(X, y)
        X_pred = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        y_pred, y_std = gp.predict(X_pred, return_std=True)

        fig, ax = _initialize_plot(save_path, figsize=(8, 6))
        sns.scatterplot(x=X.flatten(), y=y, ax=ax, color=dot_color, edgecolor='black', alpha=0.9, s=40, label='Data Points')
        ax.plot(X_pred, y_pred, color=line_color, linewidth=2, label='GP Regression')
        ax.fill_between(X_pred.flatten(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, color='#a347d1', alpha=0.2, label='95% Confidence Interval')
        min_val = min(X.min(), y.min())
        max_val = max(X.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1, label='Identity Line (y = x)')

        if icc_info and col in icc_info:
            icc_value = icc_info[col]['median_icc']
            ci_lower = icc_info[col]['ci_lower']
            ci_upper = icc_info[col]['ci_upper']
            icc_text = f"ICC = {icc_value:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
        else:
            icc_text = "ICC = N/A"

        ax.text(0.05, 0.95, f"Spearman's Rho = {spearman_rho:.3f}, {spearman_p_text}\n{icc_text}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
        
        plot_title = f'{dataset_name}:\n{col.capitalize()} ({biomarker})' if dataset_name and biomarker else f'Regression Comparison Plot for {col.capitalize()}'
        ax.set_title(plot_title, fontsize=16)
        if units:
            x_label = f'Automated method ({units})'
            y_label = f'Manual annotation ({units})'
        else:
            x_label = 'Automated method'
            y_label = 'Manual annotation'
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc='lower right')
        
        _finalize_and_save_plot(fig, f'{col}_regression_comparison_gp', save_path, dpi, show=False)
        print(f"Regression comparison plot generated for '{col}' using Gaussian Process Regression.")


def plot_bland_altman_multiple_nonparametric_subject_level(df_pred, df_gt, columns, subject_column='SubjectID', save_path='.', dataset_name=None, biomarker=None, units=None, confidence=0.95, show_legend=False):
    os.makedirs(save_path, exist_ok=True)

    for col in columns:
        df_merged = pd.merge(
            df_pred[[subject_column, col]],
            df_gt[[subject_column, col]],
            on=subject_column,
            suffixes=('_pred', '_gt')
        )
        df_merged = df_merged.dropna(subset=[f'{col}_pred', f'{col}_gt'])
        if df_merged.empty:
            print(f"No data available for column '{col}'. Skipping plot.")
            continue

        x = df_merged[f'{col}_pred']
        y = df_merged[f'{col}_gt']
        diff = x - y
        avg = (x + y) / 2

        median_diff = np.median(diff)
        lower_limit = np.percentile(diff, 2.5)
        upper_limit = np.percentile(diff, 97.5)

        boot_iterations = 10000
        boot_median = []
        boot_lower = []
        boot_upper = []
        np.random.seed(0)
        print(np.min(np.abs(diff)))  # For debugging

        for _ in range(boot_iterations):
            sample = np.random.choice(diff, size=len(diff), replace=True)
            boot_median.append(np.median(sample))
            boot_lower.append(np.percentile(sample, 2.5))
            boot_upper.append(np.percentile(sample, 97.5))

        median_ci = np.percentile(boot_median, [(1 - confidence)/2 * 100, (1 + confidence)/2 * 100])
        lower_ci = np.percentile(boot_lower, [(1 - confidence)/2 * 100, (1 + confidence)/2 * 100])
        upper_ci = np.percentile(boot_upper, [(1 - confidence)/2 * 100, (1 + confidence)/2 * 100])

        fig, ax = _initialize_plot(save_path, figsize=(8, 6))
        ax.scatter(avg, diff, color='#a347d1', edgecolor='black', alpha=0.6, s=40, label='Data Points')
        ax.axhline(median_diff, color='black', linestyle='-', linewidth=2, label=f'Median Difference ({median_diff:.2f})')
        ax.axhline(lower_limit, color='black', linestyle=':', linewidth=1.5, label=f'2.5th Percentile ({lower_limit:.2f})')
        ax.axhline(upper_limit, color='black', linestyle=':', linewidth=1.5, label=f'97.5th Percentile ({upper_limit:.2f})')
        ax.axhspan(median_ci[0], median_ci[1], color='gray', alpha=0.2, label=f'{int(confidence*100)}% CI Median')
        ax.axhspan(lower_ci[0], lower_ci[1], color='#a347d1', alpha=0.2, label=f'{int(confidence*100)}% CI Lower Limit')
        ax.axhspan(upper_ci[0], upper_ci[1], color='#a347d1', alpha=0.2, label=f'{int(confidence*100)}% CI Upper Limit')

        plot_title = f'{dataset_name}:\n{col.capitalize()} ({biomarker})' if dataset_name and biomarker else f'Bland-Altman Plot for {col.capitalize()}'
        ax.set_title(plot_title, fontsize=16)
        if units:
            x_label = f'Mean of Automated Method and Manual Annotation ({units})'
            y_label = f'Difference (Automated - Manual) ({units})'
        else:
            x_label = 'Mean of Automated Method and Manual Annotation'
            y_label = 'Difference (Automated - Manual)'
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc='upper left', fontsize=10, framealpha=0.9)
        
        loa_range = upper_limit - lower_limit
        offset = (loa_range / 100.0) * 1.5
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        xloc = 0.98
        ax.text(xloc, median_diff + offset, "Median", ha="right", va="bottom", transform=trans, fontsize=12)
        ax.text(xloc, median_diff - offset, f"{median_diff:.2f}", ha="right", va="top", transform=trans, fontsize=12)
        ax.text(xloc, lower_limit + offset, f"-1.96 SD", ha="right", va="bottom", transform=trans, fontsize=12)
        ax.text(xloc, lower_limit - offset, f"{lower_limit:.2f}", ha="right", va="top", transform=trans, fontsize=12)
        ax.text(xloc, upper_limit + offset, f"+1.96 SD", ha="right", va="bottom", transform=trans, fontsize=12)
        ax.text(xloc, upper_limit - offset, f"{upper_limit:.2f}", ha="right", va="top", transform=trans, fontsize=12)
        
        plt.grid(False)
        _finalize_and_save_plot(fig, f'{col}_bland_altman_nonparametric', save_path, 300, show=False)
        print(f"Bland-Altman plot generated for '{col}'.")








