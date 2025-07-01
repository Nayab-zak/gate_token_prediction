import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import joblib
from agents.modeling_agent import ModelingAgent
from agents.config import PREPROCESSED_FEATURES_PATH, BEST_PARAMS_PATH, MODEL_PATH

# Use already engineered features
FEATURES_PATH = PREPROCESSED_FEATURES_PATH
BEST_PARAMS_PATH = BEST_PARAMS_PATH
MODEL_PATH = MODEL_PATH

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
if not os.path.exists(BEST_PARAMS_PATH):
    raise FileNotFoundError(f"Best params file not found: {BEST_PARAMS_PATH}")

df = pd.read_csv(FEATURES_PATH, parse_dates=['MoveDate'])

# Load best parameters
import json
with open(BEST_PARAMS_PATH, 'r') as f:
    best_params = json.load(f)

# Train model using best parameters (no hyperparameter tuning)
agent = ModelingAgent(df, target='TokenCount')
model = agent.train_trees(params=best_params, tune=False)

# Save the trained model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"[optimized_train.py] Model trained with best params and saved to {MODEL_PATH}")
