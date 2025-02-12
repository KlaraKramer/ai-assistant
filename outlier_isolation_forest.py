from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# import sys
# import os

# # Add locally cloned Lux source code to path, and import Lux from there
# sys.path.insert(0, os.path.abspath('./lux'))
# import lux

def train_isolation_forest(data, contamination=0.2, intent=[]):
    # Make a copy of the data to retain original categorical labels
    data_original = data.copy()

    # Convert LuxDataFrame to Pandas DataFrame if necessary
    if not isinstance(data_original, pd.DataFrame):
        data_original = pd.DataFrame(data_original)

    # Convert object columns to numerical for model training
    data_converted = data_original.copy()
    object_cols = data_converted.select_dtypes(include=['object']).columns

    # Check if column is datetime column or categorical column
    for col in object_cols:
        try:
            parsed_col = pd.to_datetime(col, errors='coerce', infer_datetime_format=True)
            timestamp_ratio = parsed_col.notna().mean()  # Proportion of successfully converted values
            if timestamp_ratio > 0.9:  # If most values convert successfully, treat it as a timestamp
                # Convert timestamp column to integer
                year = [int(year_str[:4]) for year_str in data_converted[col]]
                data_converted[col] = year
        except Exception:
            # Treat it as a categorical column
            le = LabelEncoder()
            data_converted[col] = le.fit_transform(data_converted[col])  # Encode categories numerically

    # Train the detection model
    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(data_converted)

    # Apply predictions to the original DataFrame
    data_original['Predicted Outlier'] = yhat == -1

    # Count outliers
    outlier_count = data_original['Predicted Outlier'].sum()

    # # Convert back to LuxDataFrame
    # data_original = LuxDataFrame(data_original)

    # Preserve intent metadata if applicable
    data_original.intent = intent  

    return data_original, outlier_count
