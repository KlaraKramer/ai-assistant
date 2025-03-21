import pytest
import pandas as pd
import sys
import os

# Add the parent directory (ai-assistant/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper_functions import *

@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5, 6],
        'str': ['apple', 'banana', 'cherry', 'banana', 'date', 'elderberry', 'fig'],
        'flt': [1.0, 2.5, 3.8, 2.5, 5.9, 3.1, 0],
        'int': [100, 200, 300, 200, -2, 400, 250],
        'outlier': [False, False, False, False, True, False, True]
    })


def test_extract_intent():
    # Test normal behaviour
    output = extract_intent(['item 1', 'item 2, side note'])
    assert len(output) == 2

    # Test fallback behaviour for empty input list
    output = extract_intent([])
    assert output == []


def test_determine_contamination():
    # Test normal behaviour for more contamination
    cont_history = [0.2, 0.1, 0.15]
    output = determine_contamination(cont_history, True)
    assert output <= 0.5 and output > 0

    # Test normal behaviour for less contamination
    cont_history = [0.2, 0.3]
    output = determine_contamination(cont_history, False)
    assert output <= 0.5 and output > 0

    # Test fallback behaviour for no contamination history
    cont_history = []
    output = determine_contamination(cont_history, False)
    assert output <= 0.5 and output > 0


def test_determine_filename():
    filename = determine_filename('penguin_data.csv')
    assert filename == 'penguin_data_clean.csv'
    filename = determine_filename('corrupted_penguin.csv')
    assert filename == 'clean_penguin.csv'
    filename = determine_filename('penguincorrupted.csv')
    assert filename == 'penguinclean.csv'


def test_downloadable_data(outlier_df):
    output_df = downloadable_data(outlier_df)
    assert 'outlier' not in output_df.columns
