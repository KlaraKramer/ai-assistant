import pytest
import pandas as pd
import sys
import os

# Add the parent directory (ai-assistant/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper_functions import *


# @pytest.fixture
# def number_a():
#     return 2

# @pytest.fixture
# def number_b():
#     return 3


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
