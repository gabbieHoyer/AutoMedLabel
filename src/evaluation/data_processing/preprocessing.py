
# src/evaluation/data_processing/preprocessing.py
import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

# If you have a constant definition for ordinal_categories, you can include it here.
ordinal_categories = [['0','5', '10', '20', '40', 'full'], [0, 1, 2, 3]]

# Reorder columns function
def reorder_columns(df, all_relevant_columns, id_columns, target_columns):
    reordered_columns = id_columns + all_relevant_columns + target_columns + \
                        [col for col in df.columns if col not in id_columns + all_relevant_columns + target_columns]
    return df[reordered_columns]

def encode_and_impute(df, group_col, categorical_nominal, categorical_ordinal, continuous_numerical, discrete_numerical, id_columns, target_columns):
    all_relevant_columns = categorical_nominal + categorical_ordinal + continuous_numerical + discrete_numerical
    df = reorder_columns(df, all_relevant_columns, id_columns, target_columns)

    # Initialize encoders and imputers
    ordinal_encoder = OrdinalEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')  #sparse=False,

    # Encoding categorical columns
    df_categorical_nominal_encoded = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_nominal]),
                                                  index=df.index,
                                                  columns=onehot_encoder.get_feature_names_out(categorical_nominal))
    df_categorical_ordinal_encoded = pd.DataFrame(ordinal_encoder.fit_transform(df[categorical_ordinal]),
                                                  index=df.index,
                                                  columns=categorical_ordinal)

    # Replace original categorical columns with encoded columns
    df = pd.concat([df.drop(categorical_nominal + categorical_ordinal, axis=1),
                    df_categorical_nominal_encoded, df_categorical_ordinal_encoded], axis=1)

    # Impute continuous and discrete numerical columns
    imputer_cont = IterativeImputer(estimator=RandomForestRegressor(), initial_strategy='median', max_iter=40, random_state=0)
    imputer_disc = IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent', max_iter=40, random_state=0)

    # Impute data
    df[continuous_numerical] = imputer_cont.fit_transform(df[continuous_numerical])
    df[discrete_numerical] = imputer_disc.fit_transform(df[discrete_numerical])

    # Round imputed values for discrete columns
    for col in discrete_numerical:
        df[col] = df[col].round().astype(int)

    return df.sort_index()

def encode_impute_scale(df, categorical_nominal, categorical_ordinal, continuous_numerical, discrete_numerical, id_columns, target_columns):
    all_relevant_columns = categorical_nominal + categorical_ordinal + continuous_numerical + discrete_numerical
    df = reorder_columns(df, all_relevant_columns, id_columns, target_columns)

    # Initialize encoders, imputers, and scalers
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
    scaler_cont = StandardScaler()
    scaler_disc = MinMaxScaler()  # Using MinMaxScaler for discrete variables

    # Fit the encoders on the training data
    onehot_encoder.fit(df[categorical_nominal])
    ordinal_encoder.fit(df[categorical_ordinal])

    imputer_cont = IterativeImputer(estimator=RandomForestRegressor(), initial_strategy='median', max_iter=40, random_state=0)
    imputer_disc = IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent', max_iter=40, random_state=0)

    # Encode categorical columns separately
    encoded_cat_nominal_train = onehot_encoder.transform(df[categorical_nominal])
    encoded_cat_ordinal_train = ordinal_encoder.transform(df[categorical_ordinal])

    # Impute numerical columns
    df[continuous_numerical] = imputer_cont.fit_transform(df[continuous_numerical])
    df[discrete_numerical] = imputer_disc.fit_transform(df[discrete_numerical])

    # Apply scaling to continuous and appropriate discrete data
    df[continuous_numerical] = scaler_cont.fit_transform(df[continuous_numerical])

    # Scale only those discrete variables that benefit from it
    discrete_to_scale = ['Age','flip_angle',  'rows', 'Num_Labels']  # Add more if necessary'ETL',
    if discrete_to_scale:
        df[discrete_to_scale] = scaler_disc.fit_transform(df[discrete_to_scale])

    # Combine encoded categorical columns back with the DataFrame
    encoded_cat_nominal_df = pd.DataFrame(encoded_cat_nominal_train, index=df.index, columns=onehot_encoder.get_feature_names_out(categorical_nominal))
    encoded_cat_ordinal_df = pd.DataFrame(encoded_cat_ordinal_train, index=df.index, columns=categorical_ordinal)

    df = pd.concat([df[id_columns], encoded_cat_nominal_df, encoded_cat_ordinal_df, df[continuous_numerical + discrete_numerical], df[target_columns]], axis=1)

    return df.sort_index()