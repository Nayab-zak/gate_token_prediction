import pandas as pd
import numpy as np
from agents.preprocessing_agent import PreprocessingAgent

def test_parse_dates():
    df = pd.DataFrame({'MoveDate': ['2024-01-01', '2024-01-02']})
    agent = PreprocessingAgent()
    out = agent.parse_dates(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(out['MoveDate'])

def test_normalize_numeric():
    df = pd.DataFrame({'TokenCount': [1, 2, 3], 'ContainerCount': [2, 4, 6]})
    agent = PreprocessingAgent()
    out = agent.normalize_numeric(df.copy())
    assert out['TokenCount'].max() <= 1 and out['TokenCount'].min() >= 0

def test_handle_outliers():
    df = pd.DataFrame({'TokenCount': [1, 2, 100]})
    agent = PreprocessingAgent()
    out = agent.handle_outliers(df.copy(), cols=['TokenCount'], method='clip', cap_value=99)
    assert out['TokenCount'].max() < 100

def test_encode_missing():
    df = pd.DataFrame({'TokenCount': [1, np.nan, 3], 'Cat': [None, 'a', 'b']})
    agent = PreprocessingAgent()
    out = agent.encode_missing(df.copy())
    assert out['TokenCount'].isna().sum() == 0
    assert (out['Cat'] == 'unknown').sum() >= 1
