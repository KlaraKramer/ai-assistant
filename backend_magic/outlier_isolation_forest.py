################################################################################
### This file handles outlier detection by training an IsolationForest model ###
################################################################################

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
    category=UserWarning,
)

def train_isolation_forest(data, contamination=0.2, intent=[]):
    # Ensure valid contamination value
    if contamination <= 0.0 or contamination > 0.5:
        contamination = 0.2

    # Make a copy of the data to retain original categorical labels
    data_original = data.copy()

    # Convert LuxDataFrame to Pandas DataFrame if necessary
    if not isinstance(data_original, pd.DataFrame):
        data_original = pd.DataFrame(data_original)
    data_converted = data_original.copy()

    # Convert datetime columns to timestamps
    for col in data_converted.select_dtypes(include=['datetime64']).columns:
        data_converted[col] = data_converted[col].astype('int64') // 10**9  # Convert to seconds

    # Convert boolean columns to integers
    for col in data_converted.select_dtypes(include=['bool']).columns:
        data_converted[col] = data_converted[col].astype(int)

    # Convert object columns to numerical for model training
    object_cols = data_converted.select_dtypes(include=['object']).columns

    # Check if column is datetime column or categorical column
    for col in object_cols:
        try:
            parsed_col = pd.to_datetime(col, errors='coerce')
            timestamp_ratio = parsed_col.notna().mean()  # Proportion of successfully converted values
            if timestamp_ratio > 0.9:  # If most values convert successfully, treat it as a timestamp
                # Convert timestamp column to integer
                year = [int(year_str[:4]) for year_str in data_converted[col]]
                data_converted[col] = year
        except Exception:
            # Treat it as a categorical column
            le = LabelEncoder()
            # Encode categories numerically
            data_converted[col] = le.fit_transform(data_converted[col])  

    # Train the detection model
    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(data_converted)

    # Apply predictions to the original DataFrame
    data_original['outlier'] = yhat == -1
    # Count outliers
    outlier_count = data_original['outlier'].sum()
    # Preserve intent if applicable
    data_original.intent = intent  

    return data_original, outlier_count
