# Agent for training XGBoost model
import os
import logging
import json
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from config import DATA_DIR, MODEL_DIR, XGB_CLASSIC_ITERATIONS


def setup_logger():
    logger = logging.getLogger('06_train_xgb_classic_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_xgb_classic_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    """Load both training and validation data for proper model evaluation"""
    # Load training data
    train_path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_classic.csv')
    train_df = pd.read_csv(train_path, parse_dates=['datetime'])
    X_train = train_df.drop(columns=['datetime', 'TokenCount'])
    y_train = train_df['TokenCount']
    
    # Load validation data
    val_path = os.path.join(DATA_DIR, 'encoded_input', 'validation_input_classic.csv')
    val_df = pd.read_csv(val_path, parse_dates=['datetime'])
    X_val = val_df.drop(columns=['datetime', 'TokenCount'])
    y_val = val_df['TokenCount']
    
    return X_train, y_train, X_val, y_val


def load_hyperparams():
    model_dir = os.path.join(MODEL_DIR, 'xgb_classic')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        # default hyperparameters using iterations from config
        params = {
            'n_estimators': XGB_CLASSIC_ITERATIONS,  # Use value from config.py
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 20  # Added early stopping
        }
    return params, model_dir


def save_model_and_params(model, params, model_dir, logger, metrics=None):
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved trained XGB model to {model_path}")
    
    # Include validation metrics in hyperparameters
    if metrics:
        params['validation_metrics'] = metrics
    
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters and metrics to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting XGBoost (classic) training with validation...")

    # Load both training and validation data
    X_train, y_train, X_val, y_val = load_data()
    logger.info(f"Loaded training data X:{X_train.shape}, y:{y_train.shape}")
    logger.info(f"Loaded validation data X:{X_val.shape}, y:{y_val.shape}")

    params, model_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    # Use GPU for XGBoost if available (XGBoost >=2.0)
    params['tree_method'] = 'hist'
    params['device'] = 'cuda'
    
    # Extract early_stopping_rounds from params as it's not a model parameter
    early_stopping_rounds = params.pop('early_stopping_rounds', 20)
    
    # Create XGBoost model
    model = XGBRegressor(**params)
    
    # Train with evaluation on validation set and early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Use explicit validation set for evaluation
        eval_metric='rmse',
        early_stopping_rounds=early_stopping_rounds,
        verbose=True
    )
    
    # Get best iteration from model
    best_iteration = model.best_iteration
    logger.info(f"Best iteration: {best_iteration}")
    
    # Make predictions on validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate validation metrics
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
        'val_r2': float(val_r2),
        'best_iteration': int(best_iteration)
    }
    
    logger.info("XGBoost training completed.")

    # Save model with validation metrics
    save_model_and_params(model, params, model_dir, logger, metrics)
    logger.info("Training agent finished successfully with validation metrics.")


if __name__ == '__main__':
    main()
