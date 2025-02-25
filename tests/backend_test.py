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
def df():
    return pd.DataFrame({
        'id': [0, 1, 2, 3],
        'str': ['apple', 'banana', 'cherry', 'banana'],
        'flt': [1.0, 2.5, 3.8, 2.5],
        'int': [100, 200, 300, 200]
    })


def test_detect_duplicates(df):
    # Test normal behaviour
    output_df, dups_count = detect_duplicates(df)
    assert dups_count == 1
