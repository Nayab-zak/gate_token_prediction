# Agent for training LSTM model
import os
import logging
import json
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, MODEL_DIR, LSTM_CLASSIC_EPOCHS, USE_HYPERPARAMS_FROM_JSON


def setup_logger():
    logger = logging.getLogger('06_train_lstm_classic_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_lstm_classic_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    """Load both training and validation data for model training and evaluation"""
    # Load training data
    train_path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_classic.csv')
    train_df = pd.read_csv(train_path, parse_dates=['datetime'])
    X_train = train_df.drop(columns=['datetime', 'TokenCount']).values
    y_train = train_df['TokenCount'].values
    
    # Load validation data
    val_path = os.path.join(DATA_DIR, 'encoded_input', 'validation_input_classic.csv')
    val_df = pd.read_csv(val_path, parse_dates=['datetime'])
    X_val = val_df.drop(columns=['datetime', 'TokenCount']).values
    y_val = val_df['TokenCount'].values
    
    return X_train, y_train, X_val, y_val


def load_hyperparams(logger):
    model_dir = os.path.join(MODEL_DIR, 'lstm_classic')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')

    if USE_HYPERPARAMS_FROM_JSON and os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded hyperparameters from {hp_path}")
    else:
        params = {
            'units': 50,
            'epochs': LSTM_CLASSIC_EPOCHS,  # Use value from config.py
            'batch_size': 32,
            'patience': 10
        }
        if USE_HYPERPARAMS_FROM_JSON:
            logger.warning(f"Hyperparameter file not found at {hp_path}. Using values from config.py.")
        else:
            logger.info("Using hyperparameters directly from config.py.")

    return params, model_dir


def save_model_and_params(model, params, model_dir, logger, metrics=None):
    # Save Keras model
    model_path = os.path.join(model_dir, 'trained_model.h5')
    model.save(model_path)
    logger.info(f"Saved trained LSTM model to {model_path}")
    
    # Include validation metrics in hyperparameters
    if metrics:
        params['validation_metrics'] = metrics
    
    # Save hyperparams with metrics
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters and metrics to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting LSTM (classic) training with validation...")

    # Load both training and validation data
    X_train, y_train, X_val, y_val = load_data()
    logger.info(f"Loaded training data X:{X_train.shape}, y:{y_train.shape}")
    logger.info(f"Loaded validation data X:{X_val.shape}, y:{y_val.shape}")

    # Scale data - fit on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

    params, model_dir = load_hyperparams(logger)
    logger.info(f"Using hyperparameters: {params}")

    # Build model
    model = Sequential([
        LSTM(params['units'], input_shape=(1, X_train_scaled.shape[1])),
        Dense(1)
    ])
    
    # Add gradient clipping to the optimizer
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mae', 'mse'])

    # Check for GPU compatibility and log GPU details
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Num GPUs Available: {len(gpus)}")
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Enabled mixed precision for faster training.")
        except Exception as e:
            logger.warning(f"Could not set mixed precision: {e}")
    else:
        logger.warning("No GPU detected. Training will use CPU.")

    # Use early stopping with the explicit validation data
    early_stop = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
    
    # Train with explicit validation data instead of validation_split
    history = model.fit(
        X_train_lstm, y_train,
        epochs=params['epochs'], batch_size=params['batch_size'],
        validation_data=(X_val_lstm, y_val),  # Use explicit validation data
        callbacks=[early_stop], verbose=1
    )

    # Get validation metrics from final model
    val_metrics = model.evaluate(X_val_lstm, y_val, verbose=0)
    val_loss = val_metrics[0]
    val_mae = val_metrics[1]
    val_mse = val_metrics[2]
    val_rmse = np.sqrt(val_mse)
    
    # Log validation metrics
    logger.info(f"Validation loss: {val_loss:.4f}")
    logger.info(f"Validation MAE: {val_mae:.4f}")
    logger.info(f"Validation MSE: {val_mse:.4f}")
    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    
    # Store validation metrics
    metrics = {
        'val_loss': float(val_loss),
        'val_mae': float(val_mae),
        'val_mse': float(val_mse),
        'val_rmse': float(val_rmse)
    }

    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    save_model_and_params(model, params, model_dir, logger, metrics)
    logger.info("LSTM training completed successfully with validation metrics.")


if __name__ == '__main__':
    main()
