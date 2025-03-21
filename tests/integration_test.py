import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import *
from missing_value_detection import *
from duplicate_detection import *
from outlier_isolation_forest import *

# Mock global dataframe variable
mock_current_df = pd.DataFrame({
    'id': [0, np.nan, 2, 3, 4, 5, 6],
    'str': ['apple', 'banana', 'cherry', 'banana', 'date', 'elderberry', 'fig'],
    'flt': [1.0, 2.5, 3.8, 2.5, 5.9, 3.1, np.nan],
    'int': [100, 200, 300, 200, -2, 400, 250],
    'sec_int': [101, 202, np.nan, 202, -2, 404, 252]
})


def extract_text_from_dash_component(component):
    if isinstance(component, str):
        return component
    elif isinstance(component, list):
        return ' '.join(extract_text_from_dash_component(c) for c in component)
    elif hasattr(component, 'children'):
        return extract_text_from_dash_component(component.children)
    return ''


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_missing_value_stage():
    with patch('app.current_df', mock_current_df):
        # Test initial render of missing values
        output_div = render_missing_values(1)
        missing_df, missing_val = detect_missing_values(mock_current_df)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert f'{missing_val} missing values were detected' in output_text

        # Test displaying of rows with missing values
        output_div = update_missing_values(['highlight'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Show rows with missing values' in output_text

        # Test imputing missing values
        output_div = update_missing_values(['impute-simple'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Impute missing values using the univariate mean' in output_text
        assert '0 missing values were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_duplicate_removal_stage():
    with patch('app.current_df', mock_current_df):
        # Test initial render of duplicated rows
        output_div = render_duplicates(1)
        df, dups_count = detect_duplicates(mock_current_df)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert f'{dups_count} duplicated rows were detected' in output_text

    with patch('app.current_df', df):
        # Test displaying of duplicated rows
        output_div = update_duplicates(['highlight'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Highlight duplicated rows' in output_text

    with patch('app.current_df', df):
        # Test removal of duplicated rows
        output_div = update_duplicates(['delete'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: ' in output_text
        assert f'0 duplicated rows were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_outlier_handling_stage():
    with patch('app.current_df', mock_current_df):
        # Test initial render of outlier values
        output_div, n_clicks = render_outliers(1, None)
        df, out_count = train_isolation_forest(mock_current_df)
        assert out_count == 2
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert '2 outlier values were detected' in output_text
        assert 'outlier' in df.columns
        assert n_clicks == 1

        # Test removal of outlier values
        output_div = update_outliers(['accept-0'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Remove the detected outliers' in output_text
        assert '1 outlier values were detected' in output_text

    with patch('app.current_df', df):
        # Test detection a greater amount of outliers
        output_div = update_outliers(['more'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Find more outliers' in output_text
        assert '3 outlier values were detected' in output_text

    with patch('app.current_df', df):
        # Test detecting a smaller amount of outliers
        output_div = update_outliers(['less'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Find less outliers' in output_text
        assert '2 outlier values were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_outlier_handling_2_stage():
    df, out_count = train_isolation_forest(mock_current_df)
    assert 'outlier' in df.columns
    assert out_count == 2
    with patch('app.current_df', df):
        # Test removal of new potential outliers
        with patch('app.stage', 'outlier-handling-2'):
            output_div = update_outliers_2(['less', 'accept'], 1)
            assert output_div is not None
            output_text = extract_text_from_dash_component(output_div)
            assert 'Selected action: Remove the detected outliers' in output_text
            assert '1 new potential outlier values were detected' in output_text

            # Test detecting a greater amount of new potential outliers
            output_div = update_outliers_2(['more', 'more-2'], 1)
            assert output_div is not None
            output_text = extract_text_from_dash_component(output_div)
            assert 'Selected action: Find more outliers' in output_text
            assert '2 new potential outlier values were detected' in output_text
