import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import pandas as pd
from agents.evaluation_agent import EvaluationAgent
from utils.feature_serializer import load_features
from agents.config import MODEL_PATH, MODEL_FEATURES_PATH, PREPROCESSED_FEATURES_PATH, REPORTS_DIR, EXPERIMENT_NAME

# 1. Load model and hold-out set
def main():
    model_path = MODEL_PATH
    features_path = MODEL_FEATURES_PATH
    holdout_path = PREPROCESSED_FEATURES_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"Hold-out set not found: {holdout_path}")
    model = joblib.load(model_path)
    df_holdout = pd.read_csv(holdout_path, parse_dates=['MoveDate'])

    # Load feature names used for training
    if os.path.exists(features_path):
        import json
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
    else:
        feature_names = None

    # 2. Instantiate EvaluationAgent
    agent = EvaluationAgent(model, df_holdout, feature_names=feature_names)

    # 3. Run evaluation steps
    metrics = agent.score_holdout()
    slice_df = agent.score_slice()
    rolling_df = agent.rolling_test()

    # --- Save predictions for dashboard comparison ---
    # Predict on holdout set and save predictions.csv in reports/exp1/ (or similar)
    # You may want to set exp_name dynamically if running multiple experiments
    exp_name = EXPERIMENT_NAME  # Change as needed or make this a CLI argument
    exp_dir = os.path.join(REPORTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    df_holdout = agent.df_holdout.copy()
    X = df_holdout.drop(columns=['TokenCount', 'MoveDate'], errors='ignore')
    if agent.feature_names is not None:
        for col in agent.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[agent.feature_names]
    y_pred = agent.model.predict(X)
    df_holdout['y_pred'] = y_pred
    # Only save rows with valid MoveDate, year, and month
    df_to_save = df_holdout.dropna(subset=['MoveDate', 'year', 'month'])
    cols_to_save = [c for c in ['MoveDate', 'TokenCount', 'y_pred', 'year', 'month'] if c in df_to_save.columns]
    df_to_save[cols_to_save].to_csv(os.path.join(exp_dir, 'predictions.csv'), index=False)
    print(f"Saved filtered predictions to {os.path.join(exp_dir, 'predictions.csv')}")
    agent.report(path='reports/final_report.md')
    print('Evaluation complete. Metrics:', metrics)
    print('Slice breakdown saved. Rolling backtest and drift detection complete.')

if __name__ == '__main__':
    main()
