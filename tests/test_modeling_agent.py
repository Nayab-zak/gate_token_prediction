import pandas as pd
import numpy as np
from agents.modeling_agent import ModelingAgent

def test_train_baseline():
    df = pd.DataFrame({
        'MoveDate': pd.date_range('2024-01-01', periods=10, freq='H'),
        'TokenCount': np.arange(10),
        'ContainerCount': np.arange(10)
    })
    agent = ModelingAgent(df, target='TokenCount')
    preds = agent.train_baseline()
    assert 'hourly_mean_pred' in preds.columns
    assert 'last_week_pred' in preds.columns

def test_train_trees():
    df = pd.DataFrame({
        'MoveDate': pd.date_range('2024-01-01', periods=20, freq='H'),
        'TokenCount': np.arange(20),
        'ContainerCount': np.arange(20)
    })
    agent = ModelingAgent(df, target='TokenCount')
    agent.train_trees()
    assert 'lgbm' in agent.models or 'xgbm' in agent.models
