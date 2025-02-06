def detect_duplicates(df):
    df_copy = df
    # Remove interfering columns
    if "Id" in df_copy.columns:
        df_copy = df_copy.drop("Id", axis=1)
    if "Unnamed: 0" in df_copy.columns:
        df_copy = df_copy.drop("Unnamed: 0", axis=1)
    # Add a new column 
    df["Duplicate"] = df_copy.duplicated()
    return df

