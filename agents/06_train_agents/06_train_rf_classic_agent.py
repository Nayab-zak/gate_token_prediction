# Agent for training Random Forest model
import os
import sys
import logging
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path to fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_DIR, MODEL_DIR, RF_CLASSIC_ESTIMATORS


def setup_logger():
    logger = logging.getLogger('06_train_rf_classic_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_rf_classic_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    """
    Load both training and validation data for proper model evaluation
    """
    # Load classic encoded training data
    train_path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_classic.csv')
    val_path = os.path.join(DATA_DIR, 'encoded_input', 'validation_input_classic.csv')
    
    train_df = pd.read_csv(train_path, parse_dates=['datetime'])
    X_train = train_df.drop(columns=['datetime', 'TokenCount'])
    y_train = train_df['TokenCount']
    
    # Load validation data
    val_df = pd.read_csv(val_path, parse_dates=['datetime'])
    X_val = val_df.drop(columns=['datetime', 'TokenCount'])
    y_val = val_df['TokenCount']
    
    return X_train, y_train, X_val, y_val


def load_hyperparams():
    # Load hyperparameters if available, else use defaults
    hp_dir = os.path.join(MODEL_DIR, 'rf_classic')
    os.makedirs(hp_dir, exist_ok=True)
    hp_path = os.path.join(hp_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
            # Remove validation_metrics if present as it's not a model parameter
            if 'validation_metrics' in params:
                params.pop('validation_metrics')
    else:
        # default hyperparameters using estimators from config
        params = {"n_estimators": RF_CLASSIC_ESTIMATORS, "max_depth": 10, "random_state": 42}
    return params, hp_dir


def save_model_and_params(model, params, hp_dir, logger, metrics=None):
    # Save trained model and hyperparameters
    model_path = os.path.join(hp_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved trained RandomForest model to {model_path}")
    
    # Include metrics in params if provided
    if metrics:
        params['validation_metrics'] = metrics
    
    hp_path = os.path.join(hp_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters and metrics to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting RandomForest (classic) training with validation...")

    # Load data (including validation set)
    X_train, y_train, X_val, y_val = load_data()
    logger.info(f"Loaded training data X:{X_train.shape}, y:{y_train.shape}")
    logger.info(f"Loaded validation data X:{X_val.shape}, y:{y_val.shape}")

    # Load hyperparameters
    params, hp_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("RandomForest training completed.")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Log validation metrics
    logger.info(f"Validation MSE: {val_mse:.4f}")
    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    logger.info(f"Validation R²: {val_r2:.4f}")
    
    # Store validation metrics
    metrics = {
        'val_mse': float(val_mse),
        'val_rmse': float(val_rmse),
        'val_r2': float(val_r2)
    }

    # Save outputs with metrics
    save_model_and_params(model, params, hp_dir, logger, metrics)
    logger.info("Training agent finished successfully with validation metrics.")


if __name__ == '__main__':
    main()
