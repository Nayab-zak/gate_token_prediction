import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from agents.evaluation_agent import EvaluationAgent
from sklearn.linear_model import LinearRegression

def test_score_holdout():
    df = pd.DataFrame({
        'MoveDate': pd.date_range('2024-01-01', periods=10, freq='D'),
        'TokenCount': np.arange(10),
        'ContainerCount': np.arange(10),
        'is_holiday': [0]*10,
        'is_weekend': [0]*10
    })
    model = LinearRegression().fit(df[['ContainerCount']], df['TokenCount'])
    agent = EvaluationAgent(model, df)
    metrics = agent.score_holdout()
    assert 'MAE' in metrics and 'RMSE' in metrics

def test_score_slice():
    df = pd.DataFrame({
        'MoveDate': pd.date_range('2024-01-01', periods=10, freq='D'),
        'TokenCount': np.arange(10),
        'ContainerCount': np.arange(10),
        'is_holiday': [0, 1]*5,
        'is_weekend': [0, 1]*5
    })
    model = LinearRegression().fit(df[['ContainerCount']], df['TokenCount'])
    agent = EvaluationAgent(model, df)
    breakdown = agent.score_slice()
    assert 'MAE' in breakdown.columns
