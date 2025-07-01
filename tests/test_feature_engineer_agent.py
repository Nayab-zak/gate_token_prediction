import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from agents.feature_engineer_agent import FeatureEngineerAgent

def test_add_holiday_flags():
    df = pd.DataFrame({'MoveDate': pd.to_datetime(['2024-01-01', '2024-01-02'])})
    agent = FeatureEngineerAgent()
    out = agent.add_holiday_flags(df.copy(), country='UAE')
    assert 'is_holiday' in out.columns
    # Test unsupported country
    out2 = agent.add_holiday_flags(df.copy(), country='ZZZ')
    assert 'is_holiday' in out2.columns
    assert (out2['is_holiday'] == False).all()

def test_add_lagged_features():
    df = pd.DataFrame({'TokenCount': [1, 2, 3, 4]})
    agent = FeatureEngineerAgent()
    out = agent.add_lagged_features(df.copy(), cols=['TokenCount'], lags=[1])
    assert 'TokenCount_lag1' in out.columns

def test_add_rolling_features():
    df = pd.DataFrame({'TokenCount': [1, 2, 3, 4]})
    agent = FeatureEngineerAgent()
    out = agent.add_rolling_features(df.copy(), cols=['TokenCount'], windows=[2], aggs=['mean'])
    assert 'TokenCount_roll2_mean' in out.columns

def test_categorical_encoder():
    df = pd.DataFrame({'TokenCount': [1, 2, 3, 4], 'VesselType': ['A', 'A', 'B', 'B']})
    agent = FeatureEngineerAgent()
    out = agent.categorical_encoder(df.copy(), cols=['VesselType'])
    assert 'VesselType_enc' in out.columns
