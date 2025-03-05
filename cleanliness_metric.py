from duplicate_detection import *
from missing_value_detection import *

def count_outliers(df_og, df_new):
    if not df_og.columns.equals(df_new.columns):
        return ValueError("Columns do not match")
    outliers = {}
    # Select only numerical columns
    og_num = df_og.select_dtypes(include=[np.number]).columns
    new_num = df_new.select_dtypes(include=[np.number]).columns
    # Filter the dataframes to only include numerical columns
    df_og_num = df_og[og_num]
    df_new_num = df_new[new_num]
    for col in df_og_num:
        # Calculate the (min, max) range
        min_val = df_og_num[col].min()
        max_val = df_og_num[col].max()
        # Count the number of df_new values of that column that are outside the (min, max) range
        outlier_vals = df_new_num[(df_new_num[col] < min_val) | (df_new_num[col] > max_val)].shape[0]
        outliers[col] = outlier_vals
    # Sum up all values in outliers
    total_outliers = sum(outliers.values())
    return total_outliers

def determine_dirtiness(df_og, df):
    # Metric ranging from 0 (clean) to infinity (dirty), normalised by the length of the dataset
    dups_df, num_duplicates = detect_duplicates(df)
    num_outliers = count_outliers(df_og, df)
    missing_df, num_missing_values = detect_missing_values(df)
    num_rows = df.shape[0]
    return (num_duplicates + num_outliers + num_missing_values) / num_rows