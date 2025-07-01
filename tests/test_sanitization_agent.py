import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pytest
from agents.sanitization_agent import SanitizationAgent

def test_sanitization_agent_basic():
    # Construct toy DataFrame
    df = pd.DataFrame({
        'MoveDate': ['2024-01-01', None, '2024-01-03'],
        'TokenCount': [10, np.nan, -5],
        'ContainerCount': [np.nan, 2, -1],
        'Other': ['a', None, 'b']
    })
    agent = SanitizationAgent(df)
    cleaned = agent.sanitize(
        drop_threshold=0.5,
        fill_strategy={'TokenCount': 'median', 'ContainerCount': 0, 'Other': 'unknown'}
    )
    # 1. All MoveDate should be datetime
    assert pd.api.types.is_datetime64_any_dtype(cleaned['MoveDate'])
    # 2. All TokenCount and ContainerCount should be numeric and non-negative
    assert (cleaned['TokenCount'] >= 0).all()
    assert (cleaned['ContainerCount'] >= 0).all()
    # 3. No nulls in fill_strategy columns
    assert cleaned['TokenCount'].isna().sum() == 0
    assert cleaned['ContainerCount'].isna().sum() == 0
    assert cleaned['Other'].isna().sum() == 0
    # 4. Rows/columns with too many nulls are dropped
    # (with drop_threshold=0.5, only rows/cols with >50% nulls are dropped)
    # Should not have any row with more than 1 null
    assert (cleaned.isna().sum(axis=1) <= 1).all()
