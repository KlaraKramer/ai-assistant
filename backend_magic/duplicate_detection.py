#######################################################################################
### This file contains functionality for the automated detection of duplicated rows ###
#######################################################################################

def detect_duplicates(df, keep='first'):
    # Reset previous detection steps
    if 'duplicate' in df.columns:
        df = df.drop('duplicate', axis=1)
    df_copy = df
    # Remove interfering columns
    if 'id' in df_copy.columns:
        df_copy = df_copy.drop('id', axis=1)
    # Add a new column 
    df['duplicate'] = df_copy.duplicated(keep=keep)
    # Count the number of duplicates detected
    dups_count = df['duplicate'].value_counts().get(True, 0)
    return df, dups_count

