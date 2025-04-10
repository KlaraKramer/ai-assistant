################################################################################################################
### This file tests the AI-driven backend, including:                                                        ###
### - Verifying that duplicate detection and removal works correctly in normal and edge cases                ###
### - Testing the various functionalities of missing value detection, imputation, and removal methods        ###
### - Ensuring that the IsolationForest model for outlier detection works as expected and yields accurate    ###
###   results and that the outlier handling function is robust against out-of-range contamination parameters ###
################################################################################################################

import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_magic.duplicate_detection import *
from backend_magic.missing_value_detection import *
from backend_magic.outlier_isolation_forest import *


################################
### Specify testing fixtures ###

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

@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5, 6],
        'str': ['apple', 'banana', 'cherry', 'banana', 'date', 'elderberry', 'fig'],
        'flt': [1.0, 2.5, 3.8, 2.5, 5.9, 3.1, 0],
        'int': [100, 200, 300, 200, -2, 400, 250]
    })


##########################################
### Test the duplicate detection stage ###

def test_detect_duplicates(duplicate_df):
    # Test normal behaviour
    output_df, dups_count = detect_duplicates(duplicate_df)
    assert dups_count == 1
    assert output_df.shape[0] == 4

    # Test keep=False behaviour
    output_df, dups_count = detect_duplicates(duplicate_df, keep=False)
    assert dups_count == 2
    assert output_df.shape[0] == 4


def test_detect_no_duplicates(missing_df):
    # Test normal behaviour if no duplicates are present
    output_df, dups_count = detect_duplicates(missing_df)
    assert dups_count == 0
    assert output_df.shape[0] == 4

    # Test keep=False behaviour if no duplicates are present
    output_df, dups_count = detect_duplicates(missing_df, keep=False)
    assert dups_count == 0
    assert output_df.shape[0] == 4


#############################################
### Test the missing value handling stage ###

def test_detect_missing_values(missing_df, duplicate_df):
    # Test normal behaviour
    output_df, miss_count = detect_missing_values(missing_df)
    assert miss_count == 4
    assert output_df.shape == (1, 4)

    # Test behaviour if no values are missing
    output_df, miss_count = detect_missing_values(duplicate_df)
    assert miss_count == 0
    assert output_df.shape == (1, 4)


def test_detect_no_missing_values(outlier_df):
    # Test normal behaviour using a different dataframe
    output_df, miss_count = detect_missing_values(outlier_df)
    assert miss_count == 0
    assert output_df.shape == (1, 4)


def test_simple_imputation_normal(missing_df):
    # Test normal behaviour of missing value imputation
    output_df = impute_missing_values(missing_df, method='simple')
    new_df, miss_count = detect_missing_values(output_df)
    assert miss_count == 0
    assert new_df.shape[0] == 1
    assert new_df.shape[1] == 4

def test_simple_imputation_no_spec(missing_df):
    # Test behaviour without specifying the method of missing value imputation
    output_df = impute_missing_values(missing_df)
    new_df, miss_count = detect_missing_values(output_df)
    assert miss_count == 0
    assert new_df.shape == (1, 4)

    # Test imputed values
    assert output_df['id'].mean() == 1.5
    assert missing_df['flt'].mean() == output_df['flt'].mean()
    assert output_df['int'].mean() == 200


def test_knn_imputation(missing_df):
    # Test normal behaviour using the KNN imputation
    output_df = impute_missing_values(missing_df, method='KNN')
    new_df, miss_count = detect_missing_values(output_df)
    assert miss_count == 0
    assert new_df.shape == (1, 4)

    # Test imputed values
    assert output_df['id'].mean() == np.float64(1.625)
    assert missing_df['flt'].mean() != output_df['flt'].mean()
    assert output_df['int'].mean() == 200


#######################################
### Test the outlier handling stage ###

@pytest.mark.filterwarnings('ignore:Could not infer format, so each element will be parsed individually:UserWarning')
def test_outlier_detection(outlier_df):
    # Test normal behaviour of outlier detection function
    output_df, out_count = train_isolation_forest(outlier_df)
    assert out_count == 2
    assert output_df['outlier'].tolist()[1] == False

@pytest.mark.filterwarnings('ignore:Could not infer format, so each element will be parsed individually:UserWarning')
def test_outlier_contamination(outlier_df):
    # Test specification of contamination parameter
    output_df, out_count = train_isolation_forest(outlier_df, contamination=0.1)
    assert out_count == 1
    assert output_df['outlier'].tolist()[1] == False

    output_df, out_count = train_isolation_forest(outlier_df, contamination=0.5)
    assert out_count == 3
    assert output_df['outlier'].tolist()[1] == False
    assert output_df['outlier'].tolist()[4] == True

    # Test out-of-range specification of contamination parameter
    output_df, out_count = train_isolation_forest(outlier_df, contamination=0)
    assert out_count == 2
    assert output_df['outlier'].tolist()[1] == False

    output_df, out_count = train_isolation_forest(outlier_df, contamination=0.6)
    assert out_count == 2
    assert output_df['outlier'].tolist()[1] == False
