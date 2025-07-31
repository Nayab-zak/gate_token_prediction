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
from config import DATA_DIR, MODEL_DIR


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
    path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_classic.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    X = df.drop(columns=['datetime', 'TokenCount']).values
    y = df['TokenCount'].values
    return X, y


def load_hyperparams():
    model_dir = os.path.join(MODEL_DIR, 'lstm_classic')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        # default hyperparameters
        params = {
            'units': 50,
            'epochs': 50,
            'batch_size': 32,
            'patience': 10
        }
    return params, model_dir


def save_model_and_params(model, params, model_dir, logger):
    # Save Keras model
    model_path = os.path.join(model_dir, 'trained_model.h5')
    model.save(model_path)
    logger.info(f"Saved trained LSTM model to {model_path}")
    # Save hyperparams
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting LSTM (classic) training...")

    X, y = load_data()
    logger.info(f"Loaded training data X:{X.shape}, y:{y.shape}")

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Reshape for LSTM: (samples, timesteps=1, features)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    params, model_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    # Build model
    model = Sequential([
        LSTM(params['units'], input_shape=(1, X_scaled.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    early_stop = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
    model.fit(
        X_lstm, y,
        epochs=params['epochs'], batch_size=params['batch_size'],
        validation_split=0.1, callbacks=[early_stop], verbose=1
    )

    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    save_model_and_params(model, params, model_dir, logger)
    logger.info("LSTM training completed successfully.")

    # Check for GPU and enable mixed precision for TensorFlow
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
