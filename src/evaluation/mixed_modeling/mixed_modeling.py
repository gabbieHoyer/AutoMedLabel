import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def load_processed_data(filepath):
    """
    Load processed data saved as Feather or CSV.
    """
    if filepath.endswith('.feather'):
        return pd.read_feather(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format.")

def aggregate_data(df, group_columns):
    """
    Aggregate the dataframe to dataset level.
    Continuous features are averaged,
    categorical features are aggregated by mode (or first),
    and the outcome is taken as the first value.
    """
    # These lists are provided for example purposes.
    # You may allow the user to supply these lists as parameters.
    continuous_features = ['TE', 'pixel_spacing_x', 'slice_thickness', 'flip_angle', 'rows']
    categorical_features = ['twoD_threeD_threeD', 'scanner_vendor_Siemens']
    outcome_column = 'dataset_exp_global_dice'
    
    df_cont = df.groupby(group_columns)[continuous_features].mean()
    df_cat = df.groupby(group_columns)[categorical_features].agg(lambda x: pd.Series.mode(x)[0] if not x.mode().empty else None)  #.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    df_out = df.groupby(group_columns)[outcome_column].first()
    
    df_dataset = pd.concat([df_cont, df_cat, df_out], axis=1).reset_index()
    return df_dataset

def calculate_vif_df(df, target_columns=None):
    """
    Calculate VIF. Assumes that df already includes a constant.
    """
    from statsmodels.tools.tools import add_constant
    X = add_constant(df.drop(columns=target_columns)) if target_columns else add_constant(df)
    vif_df = pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_df

def run_mixed_effects_model(df, formula, group_col, output_dir):
    """
    Fit a mixed effects model on the provided dataframe.
    Saves the model summary to a text file.
    Returns the fitted model.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = smf.mixedlm(formula, data=df, groups=df[group_col])
    result = model.fit()
    # save_model_summary(result, group_col, output_dir)
    return result

def save_model_summary(result, exp_type, output_dir):
    """
    Save the model summary as a text file.
    """
    summary_path = os.path.join(output_dir, f'model_summary_exp_{exp_type}.txt')
    with open(summary_path, 'w') as f:
        f.write(result.summary().as_text())

def clean_label(variable_name):
    """
    Clean variable names for plotting.
    """
    replacements = {
        'experiment_type_new': 'ExpType',
        'twoD_threeD_threeD': '2D/3D',
        'image_encoder_True': 'ImgEnc',
        'bbox_shift_20': 'BBoxShift',
        'pixel_spacing_x': 'PixelSpacing'
    }
    for old, new in replacements.items():
        variable_name = variable_name.replace(old, new)
    variable_name = variable_name.replace(':', ' × ')
    return variable_name

def plot_fixed_effects(result, exp_type, output_dir):
    """
    Plot fixed effects estimates with 95% confidence intervals.
    """
    fe = result.fe_params
    conf_int = result.conf_int()
    if 'Group Var' in conf_int.index:
        conf_int = conf_int.drop('Group Var')
    if len(fe) != conf_int.shape[0]:
        print("Mismatch in fixed effect lengths.")
        return

    fe_df = pd.DataFrame({
        'Estimate': fe,
        'CI_lower': conf_int.iloc[:, 0],
        'CI_upper': conf_int.iloc[:, 1],
        'Variable': [clean_label(var) for var in fe.index]
    })
    fe_df.sort_values('Estimate', inplace=True, ascending=True)
    fe_df['Summary'] = [f"{est:.2f} [{cil:.2f}, {ciu:.2f}]" for est, cil, ciu in zip(fe_df['Estimate'], fe_df['CI_lower'], fe_df['CI_upper'])]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(fe_df['Estimate'], range(len(fe_df)),
                 xerr=[fe_df['Estimate'] - fe_df['CI_lower'], fe_df['CI_upper'] - fe_df['Estimate']],
                 fmt='o', ecolor='gray', capsize=5, color='#C06BeB', markersize=5)
    ax1.set_yticks(range(len(fe_df)))
    ax1.set_yticklabels(fe_df['Variable'])
    ax1.axvline(x=0, color='gray', linestyle='--')
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(range(len(fe_df)))
    ax2.set_yticklabels(fe_df['Summary'][::-1], fontdict={'fontsize': 8, 'color': 'black'})
    plt.title(f'Fixed Effects Estimates with 95% CI for Experiment Type {exp_type} vs. Baseline')
    ax1.set_xlabel('Effect Size')
    ax1.set_ylabel('Effects')
    ax2.set_ylabel('Effect Size (95% CI)')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fixed_effects_exp_{exp_type}.png'), format='png')
    fig.savefig(os.path.join(output_dir, f'fixed_effects_exp_{exp_type}.svg'), format='svg')
    plt.show()
