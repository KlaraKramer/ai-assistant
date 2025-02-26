import pytest
import pandas as pd
import sys
import os

# Add the parent directory (ai-assistant/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from duplicate_detection import *
from missing_value_detection import *
from outlier_isolation_forest import *


@pytest.fixture
def duplicate_df():
    return pd.DataFrame({
        'id': [0, 1, 2, 3],
        'str': ['apple', 'banana', 'cherry', 'banana'],
        'flt': [1.0, 2.5, 3.8, 2.5],
        'int': [100, 200, 300, 200]
    })

@pytest.fixture
def missing_df():
    return pd.DataFrame({
        'id': [np.nan, 1, 2, np.nan],
        'str': ['apple', 'banana', 'cherry', 'banana'],
        'flt': [1.0, 2.5, 3.8, np.nan],
        'int': [100, np.nan, 300, 200]
    })


def test_detect_duplicates(duplicate_df):
    # Test normal behaviour
    output_df, dups_count = detect_duplicates(duplicate_df)
    assert dups_count == 1
    assert output_df.shape[0] == 4

    # Test keep=False behaviour
    output_df, dups_count = detect_duplicates(duplicate_df, keep=False)
    assert dups_count == 2
    assert output_df.shape[0] == 4


def test_detect_missing_values(missing_df):
    # Test normal behaviour
    output_df, miss_count = detect_missing_values(missing_df)
    assert miss_count == 4
    # assert output_df.shape
