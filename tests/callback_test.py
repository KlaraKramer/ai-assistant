################################################################################
### This file tests the central callback functions for rendering components  ### 
### on the GUI within all stages of the data engineering process             ###
################################################################################

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import *
from backend_magic.missing_value_detection import *
from backend_magic.duplicate_detection import *
from backend_magic.outlier_isolation_forest import *


###################################
### Mock dataframes for testing ###

mock_current_df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4]
})

mock_duplicate_df = pd.DataFrame({
    'id': [0, 1, 2, 3, 4, 5, 6],
    'str': ['apple', 'banana', 'cherry', 'banana', 'date', 'elderberry', 'fig'],
    'flt': [1.0, 2.5, 3.8, 2.5, 5.9, 3.1, 0],
    'int': [100, 200, 300, 200, -2, 400, 250],
    'sec_int': [101, 202, 303, 202, -2, 404, 252]
})


####################################################################
### Helper functions for the testing of text rendered on the GUI ###

def extract_text_from_dash_component(component):
    # Return the text contained in a dash container as a string
    if isinstance(component, str):
        return component
    elif isinstance(component, list):
        return ' '.join(extract_text_from_dash_component(c) for c in component)
    elif hasattr(component, 'children'):
        return extract_text_from_dash_component(component.children)
    return ''


###################################################################
### Test the rendering and updating of missing values on the UI ###

def test_render_missing_values():
    # Test normal behaviour of the first render within the missing value handling stage
    with patch('app.current_df', mock_current_df):
        output_div = render_missing_values(1)
        missing_df, missing_val = detect_missing_values(mock_current_df)
        assert output_div is not None
        # Verify output text
        output_text = extract_text_from_dash_component(output_div)
        assert f'{missing_val} missing values were detected' in output_text


def test_update_missing_values():
    # Test subsequent renders of the missing value handling stage
    with patch('app.current_df', mock_current_df):
        output_div = update_missing_values(['highlight'], 1)
        assert output_div is not None
        # Verify output text
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Show rows with missing values' in output_text
    
    # Test output after missing value imputation
    with patch('app.current_df', mock_current_df):
        output_div = update_missing_values(['impute-simple'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Impute missing values using the univariate mean' in output_text
        assert '0 missing values were detected' in output_text


####################################################################
### Test the rendering and updating of duplicated rows on the UI ###

@pytest.mark.filterwarnings('ignore:Could not infer format')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_render_duplicates():
    # Test normal behaviour of the initial render within the duplicate detection stage
    with patch('app.current_df', mock_duplicate_df):
        output_div = render_duplicates(1)
        df, dups_count = detect_duplicates(mock_duplicate_df)
        assert output_div is not None
        # Verify output text displayed to users
        output_text = extract_text_from_dash_component(output_div)
        assert f'{dups_count} duplicated rows were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_update_duplicates():
    # Test behaviour of subsequent renders during the duplicate detection stage
    df, dups_count = detect_duplicates(mock_duplicate_df)
    with patch('app.current_df', df):
        output_div = update_duplicates(['delete'], 1)
        assert output_div is not None
        # Verify output text correctness
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: ' in output_text
        assert f'0 duplicated rows were detected' in output_text


#############################################################
### Test the rendering and updating of outliers on the UI ###

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_render_outliers():
    # Test normal behaviour of the initial render within the outlier handling stage
    with patch('app.current_df', mock_duplicate_df):
        output_div, n_clicks = render_outliers(1, None)
        df, out_count = train_isolation_forest(mock_duplicate_df)
        assert out_count == 2
        assert output_div is not None
        # Verify output text displayed on the GUI
        output_text = extract_text_from_dash_component(output_div)
        assert '1 outlier values were detected' in output_text
        assert 'outlier' in df.columns
        assert n_clicks == 1


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_update_outliers():
    # Test behaviour of subsequent renders within the outlier handling stage
    df, out_count = train_isolation_forest(mock_duplicate_df)
    assert 'outlier' in df.columns
    assert out_count == 2
    with patch('app.current_df', df):
        output_div = update_outliers(['accept-0'], 1)
        assert output_div is not None
        # Verify output text
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Remove the detected outliers' in output_text
        assert '1 outlier values were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_update_outliers_more():
    # Test the functionality of detecting more outliers
    df, out_count = train_isolation_forest(mock_duplicate_df)
    assert 'outlier' in df.columns
    assert out_count > 0
    with patch('app.current_df', df):
        # Simulate user input
        output_div = update_outliers(['more'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Find more outliers' in output_text
        assert '2 outlier values were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_update_outliers_less():
    # Test the functionality of detecting less outliers
    df, out_count = train_isolation_forest(mock_duplicate_df)
    assert 'outlier' in df.columns
    assert out_count > 0
    with patch('app.current_df', df):
        # Simulate user input
        output_div = update_outliers(['less'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Find less outliers' in output_text
        assert '1 outlier values were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_update_outliers_2():
    # Test behaviour of subsequent renders within the outlier handling stage
    df, out_count = train_isolation_forest(mock_duplicate_df)
    assert 'outlier' in df.columns
    assert out_count == 2
    with patch('app.current_df', df):
        with patch('app.stage', 'outlier-handling-2'):
            # Simulate history of user interactions
            output_div = update_outliers_2(['less', 'accept'], 1)
            assert output_div is not None
            # Verify output text
            output_text = extract_text_from_dash_component(output_div)
            assert 'Selected action: Remove the detected outliers' in output_text
            assert ' new potential outlier values were detected' in output_text


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_update_outliers_2_more():
    # Test behaviour of renders within the final section of the outlier handling stage
    df, out_count = train_isolation_forest(mock_duplicate_df)
    assert 'outlier' in df.columns
    assert out_count > 0
    with patch('app.current_df', df):
        with patch('app.stage', 'outlier-handling-2'):
            # Simulate history of user interactions
            output_div = update_outliers_2(['more', 'more-2'], 1)
            assert output_div is not None
            # Verify output text correctness
            output_text = extract_text_from_dash_component(output_div)
            assert 'Selected action: Find more outliers' in output_text
            assert '2 new potential outlier values were detected' in output_text
