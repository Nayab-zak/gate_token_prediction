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
from config import DATA_DIR, MODEL_DIR, LSTM_AUGMENTED_EPOCHS


def setup_logger():
    logger = logging.getLogger('06_train_lstm_augmented_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_lstm_augmented_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_augmented.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    X = df.drop(columns=['datetime', 'TokenCount']).values
    y = df['TokenCount'].values
    return X, y


def load_hyperparams():
    model_dir = os.path.join(MODEL_DIR, 'lstm_augmented')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        params = {
            'units': 50,
            'epochs': LSTM_AUGMENTED_EPOCHS,  # Use value from config.py
            'batch_size': 32,
            'patience': 10
        }
    return params, model_dir


def save_model_and_params(model, params, model_dir, logger):
    model_path = os.path.join(model_dir, 'trained_model.h5')
    model.save(model_path)
    logger.info(f"Saved trained LSTM (augmented) model to {model_path}")
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting LSTM (augmented) training...")

    X, y = load_data()
    logger.info(f"Loaded training data X:{X.shape}, y:{y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    params, model_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

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

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    save_model_and_params(model, params, model_dir, logger)
    logger.info("LSTM (augmented) training completed successfully.")


if __name__ == '__main__':
    main()
