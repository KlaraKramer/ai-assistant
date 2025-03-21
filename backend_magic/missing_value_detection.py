import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

def detect_missing_values(df):
    # Count the number of missing values detected in each column
    missing_df = df.isnull().sum()
    missing_df = pd.DataFrame(missing_df)
    missing_df = missing_df.T
    # Count the total number of missing values
    missing_val = int(missing_df.sum().sum())
    return missing_df, missing_val

def impute_missing_values(df, method='simple'):
    df_copy = df.copy()  # Avoid modifying the original DataFrame
    # Select only numeric columns
    num_cols = df_copy.select_dtypes(include=[np.number]).columns

    if method == 'simple':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    if method == 'KNN':
        imp = KNNImputer(n_neighbors=2, weights="uniform")
    else:
        # Default to the simple imputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Apply imputation only to numeric columns    
    df_copy[num_cols] = imp.fit_transform(df_copy[num_cols])
    return df_copy