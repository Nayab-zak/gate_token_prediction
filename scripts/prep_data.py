import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from agents.data_laoder_agent import DataLoaderAgent
from agents.sanitization_agent import SanitizationAgent
from agents.config import INPUT_FILE_PATH, PREPROCESSED_FEATURES_PATH

RAW_PATH = INPUT_FILE_PATH
PREP_PATH = PREPROCESSED_FEATURES_PATH

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Raw input file not found: {RAW_PATH}")

# 1. Load raw data
loader = DataLoaderAgent()
df = loader.load_from_file(RAW_PATH)

# 2. Sanitize data
sanitizer = SanitizationAgent(df)
df_clean = sanitizer.sanitize()

# 3. Save preprocessed data
os.makedirs(os.path.dirname(PREP_PATH), exist_ok=True)
df_clean.to_csv(PREP_PATH, index=False)
print(f"[prep_data.py] Preprocessed data saved to {PREP_PATH}")
