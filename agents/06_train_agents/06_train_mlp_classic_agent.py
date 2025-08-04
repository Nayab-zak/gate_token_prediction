import os
import logging
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from config import DATA_DIR, MODEL_DIR, MLP_CLASSIC_EPOCHS


def setup_logger():
    logger = logging.getLogger('06_train_mlp_classic_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_mlp_classic_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    """Load both training and validation data for model training and evaluation"""
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
    model_dir = os.path.join(MODEL_DIR, 'mlp_classic')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        # default hyperparameters using epochs from config
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': MLP_CLASSIC_EPOCHS,  # Use value from config.py
            'random_state': 42
        }
    return params, model_dir


def save_model_and_params(model, params, model_dir, logger, metrics=None):
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved trained MLP model to {model_path}")
    
    # Include validation metrics in hyperparameters
    if metrics:
        params['validation_metrics'] = metrics
        
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters and metrics to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting MLP (classic) training with validation...")

    # Load both training and validation data
    X_train, y_train, X_val, y_val = load_data()
    logger.info(f"Loaded training data X:{X_train.shape}, y:{y_train.shape}")
    logger.info(f"Loaded validation data X:{X_val.shape}, y:{y_val.shape}")

    # Scale inputs - fit on training data, transform both training and validation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Load hyperparameters
    params, model_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    # Remove unsupported parameters
    params.pop('validation_metrics', None)

    # Train model
    model = MLPRegressor(**params)
    model.fit(X_train_scaled, y_train)
    logger.info("MLP training completed.")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
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

    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Save model and hyperparameters with metrics
    save_model_and_params(model, params, model_dir, logger, metrics)
    logger.info("Training agent finished successfully with validation metrics.")

    # Check for GPU availability and set mixed precision for TensorFlow
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Num GPUs Available: {len(gpus)}")
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("Enabled mixed precision for faster training.")
        except Exception as e:
            print(f"Could not set mixed precision: {e}")
    else:
        print("No GPU detected. Training will use CPU.")


if __name__ == '__main__':
    main()
