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

@pytest.fixture
def timestamp_df():
    return pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5, 6],
        'str': ['apple', 'banana', 'cherry', 'banana', 'date', 'elderberry', 'fig'],
        'flt': [1.0, 2.5, 3.8, 2.5, 5.9, 3.1, 0],
        'int': [100, 200, 300, 200, -2, 400, 250],
        'reg': ['2021-07-03 16:21:12.357246', '2025-01-04 07:21:12.123456', '2020-03-17 15:00:00.357246', '2001-04-04 14:02:00.000000', '2025-12-06 21:15:37.428000', '2021-07-03 16:21:12.357246', '2020-03-20 19:00:00.946205']
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
    assert 'flt' in output_df.columns
    assert len(output_df.columns) == 4


def test_extract_vis_columns():
    extracted_columns = extract_vis_columns('<Vis  (x: petallengthcm, y: petalwidthcm) mark: scatter, score: 0.8385862259042764 >')
    assert extracted_columns == ['petallengthcm', 'petalwidthcm']
    extracted_columns = extract_vis_columns('<Vis  (x: annual_mileage_x1000_km, y: previous_accidents, color: duplicate) mark: scatter, score: 1.0 >')
    assert extracted_columns == ['annual_mileage_x1000_km', 'previous_accidents']


@pytest.mark.filterwarnings('ignore:Could not infer format, so each element will be parsed individually:UserWarning')
def test_parse_datetime_cols(timestamp_df):
    output_df = parse_datetime_cols(timestamp_df)
    assert 'reg' in output_df.columns
