import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
from agents.data_laoder_agent import DataLoaderAgent

def test_load_from_file_csv(tmp_path):
    # Create a dummy CSV
    df = pd.DataFrame({'MoveDate': ['2024-01-01'], 'TokenCount': [1]})
    file = tmp_path / 'test.csv'
    df.to_csv(file, index=False)
    agent = DataLoaderAgent()
    loaded = agent.load_from_file(str(file))
    assert loaded.shape == (1, 2)
    assert pd.api.types.is_datetime64_any_dtype(loaded['MoveDate'])

def test_load_from_file_xlsx(tmp_path):
    df = pd.DataFrame({'MoveDate': ['2024-01-01'], 'TokenCount': [1]})
    file = tmp_path / 'test.xlsx'
    df.to_excel(file, index=False)
    agent = DataLoaderAgent()
    loaded = agent.load_from_file(str(file))
    assert loaded.shape == (1, 2)
    assert pd.api.types.is_datetime64_any_dtype(loaded['MoveDate'])

def test_load_unknown_extension(tmp_path):
    file = tmp_path / 'test.txt'
    file.write_text('bad')
    agent = DataLoaderAgent()
    with pytest.raises(ValueError):
        agent.load_from_file(str(file))
