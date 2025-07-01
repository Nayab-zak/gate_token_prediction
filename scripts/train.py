import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.modeling_agent import ModelingAgent
from agents.config import PREPROCESSED_FEATURES_PATH, MODEL_PATH, MODEL_FEATURES_PATH

import joblib
from utils.feature_serializer import load_features
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment('token_prediction')
with mlflow.start_run(run_name='train_model'):
    mlflow.log_param('target', 'TokenCount')
    # 1. Load engineered features
    data_path = PREPROCESSED_FEATURES_PATH
    df = load_features(data_path) if data_path.endswith('.parquet') else None
    if df is None:
        import pandas as pd
        df = pd.read_csv(data_path)
    # 2. Instantiate ModelingAgent (assume target is 'TokenCount')
    agent = ModelingAgent(df, target='TokenCount')
    # 3. Train baselines and models
    agent.train_baseline()
    agent.train_glms()
    agent.train_trees()
    agent.tune_parameters()
    # 4. Save best model (choose best from agent.models, here LightGBM preferred)
    best_model = agent.models.get('lgbm') or agent.models.get('xgbm')
    if best_model is not None:
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        # Save feature names used for training (tree models)
        feature_names = agent.get_feature_names()
        with open(MODEL_FEATURES_PATH, 'w') as f:
            import json
            json.dump(feature_names, f)
        print(f"Best model and feature names saved to {MODEL_PATH} and {MODEL_FEATURES_PATH}")
        # Log model artifact
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(MODEL_FEATURES_PATH)
        # --- Log metrics ---
        if hasattr(agent, 'X_test') and hasattr(agent, 'y_test'):
            y_pred = best_model.predict(agent.X_test)
            y_true = agent.y_test
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('MAPE', mape)
            # --- Log plots ---
            plt.figure(figsize=(8,4))
            plt.plot(y_true.values, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.legend()
            plt.title('Actual vs Predicted TokenCount')
            plt.tight_layout()
            plot_path = 'models/actual_vs_predicted.png'
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            # Residuals plot
            plt.figure(figsize=(8,4))
            plt.hist(y_true - y_pred, bins=30)
            plt.title('Residuals Histogram')
            plt.tight_layout()
            resid_path = 'models/residuals_histogram.png'
            plt.savefig(resid_path)
            mlflow.log_artifact(resid_path)
            plt.close()
