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

# Mock global dataframe variables
mock_current_df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4]
})

mock_duplicate_df = pd.DataFrame({
    'id': [0, 1, 2, 3, 4, 5, 6],
    'str': ['apple', 'banana', 'cherry', 'banana', 'date', 'elderberry', 'fig'],
    'flt': [1.0, 2.5, 3.8, 2.5, 5.9, 3.1, 0],
    'int': [100, 200, 300, 200, -2, 400, 250]
})

def extract_text_from_dash_component(component):
    if isinstance(component, str):
        return component
    elif isinstance(component, list):
        return ' '.join(extract_text_from_dash_component(c) for c in component)
    elif hasattr(component, 'children'):
        return extract_text_from_dash_component(component.children)
    return ''

def test_render_missing_values():
    with patch('app.current_df', mock_current_df):
        output_div = render_missing_values(1)
        missing_df, missing_val = detect_missing_values(mock_current_df)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert f'{missing_val} missing values were detected' in output_text

def test_update_missing_values():
    with patch('app.current_df', mock_current_df):
        output_div = update_missing_values(['highlight'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Show rows with missing values' in output_text
    
    with patch('app.current_df', mock_current_df):
        output_div = update_missing_values(['impute-simple'], 1)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert 'Selected action: Impute missing values using the univariate mean' in output_text
        assert '0 missing values were detected' in output_text

@pytest.mark.filterwarnings('ignore:Could not infer format')
@pytest.mark.filterwarnings('ignore:is_period_dtype is deprecated')
@pytest.mark.filterwarnings('ignore:is_interval_dtype is deprecated')
@pytest.mark.filterwarnings('ignore:The converter attribute was deprecated')
def test_render_duplicates():
    with patch('app.current_df', mock_duplicate_df):
        output_div = render_duplicates(1)
        df, dups_count = detect_duplicates(mock_duplicate_df)
        assert output_div is not None
        output_text = extract_text_from_dash_component(output_div)
        assert f'{dups_count} duplicated rows were detected' in output_text
