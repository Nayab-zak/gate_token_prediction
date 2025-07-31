import os
import logging
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import DATA_DIR


def setup_logger():
    logger = logging.getLogger('07_test_lstm_augmented_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/07_test_lstm_augmented_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_model_and_scaler():
    model_dir = os.getenv('MODEL_DIR', 'models')
    model_path = os.path.join(model_dir, 'lstm_augmented', 'trained_model.h5')
    scaler_path = os.path.join(model_dir, 'lstm_augmented', 'scaler.pkl')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("LSTM augmented model or scaler not found")
    
    # Try to load model with GPU first, fallback to CPU if needed
    model = smart_load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def smart_load_model(model_path):
    """
    Smart model loading with GPU/CPU fallback for TensorFlow/Keras models.
    """
    logger = logging.getLogger('07_test_lstm_augmented_agent')
    
    try:
        # Try GPU first
        logger.info("Attempting to load model with GPU support...")
        with tf.device('/GPU:0'):
            model = load_model(model_path)
        logger.info("✅ Model loaded successfully with GPU support")
        return model
    except Exception as gpu_error:
        logger.warning(f"GPU model loading failed: {str(gpu_error)}")
        logger.info("🔄 Falling back to CPU model loading...")
        try:
            # Force CPU
            with tf.device('/CPU:0'):
                # Disable GPU for this session
                tf.config.set_visible_devices([], 'GPU')
                model = load_model(model_path)
            logger.info("✅ Model loaded successfully with CPU fallback")
            return model
        except Exception as cpu_error:
            logger.error(f"Both GPU and CPU model loading failed. GPU error: {gpu_error}, CPU error: {cpu_error}")
            raise RuntimeError(f"Model loading failed on both GPU and CPU. Last error: {cpu_error}")


def load_test_data():
    # Load encoded data for model prediction
    encoded_path = os.path.join(DATA_DIR, 'encoded_input', 'test_input_augmented.csv')
    df_encoded = pd.read_csv(encoded_path, parse_dates=['datetime'])
    X = df_encoded.drop(columns=['datetime', 'TokenCount']).values
    y = df_encoded['TokenCount'].values
    timestamps = df_encoded['datetime']
    
    # Load original features to get transaction-level identifiers
    features_path = os.path.join(DATA_DIR, 'features', 'test_features.csv')
    df_features = pd.read_csv(features_path, parse_dates=['datetime'])
    
    # Extract transaction identifiers (ensure same order as encoded data)
    df_features = df_features.sort_values('datetime').reset_index(drop=True)
    transaction_keys = df_features[['MoveType', 'TerminalID', 'Desig']].copy()
    
    return X, y, timestamps, transaction_keys


def smart_predict_lstm(model, X_lstm, logger):
    """
    Smart prediction with GPU/CPU fallback for LSTM models.
    """
    try:
        # Try GPU prediction first
        logger.info("Attempting LSTM prediction on GPU...")
        with tf.device('/GPU:0'):
            preds = model.predict(X_lstm).flatten()
        logger.info("✅ GPU LSTM prediction successful")
        return preds
    except Exception as gpu_error:
        logger.warning(f"GPU LSTM prediction failed: {str(gpu_error)}")
        logger.info("🔄 Falling back to CPU LSTM prediction...")
        try:
            # Force CPU prediction
            with tf.device('/CPU:0'):
                preds = model.predict(X_lstm).flatten()
            logger.info("✅ CPU LSTM fallback prediction successful")
            return preds
        except Exception as cpu_error:
            logger.error(f"Both GPU and CPU LSTM prediction failed. GPU error: {gpu_error}, CPU error: {cpu_error}")
            raise RuntimeError(f"LSTM prediction failed on both GPU and CPU. Last error: {cpu_error}")


def evaluate_and_save(model, scaler, X, y, timestamps, transaction_keys, logger):
    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Smart prediction with GPU/CPU fallback
    preds = smart_predict_lstm(model, X_lstm, logger)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    mape = np.mean(np.abs((y - preds) / y)) * 100
    logger.info(f"LSTM Augmented Test - RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

    results = pd.DataFrame({
        'datetime': timestamps,
        'TerminalID': transaction_keys['TerminalID'],
        'MoveType': transaction_keys['MoveType'], 
        'Desig': transaction_keys['Desig'],
        'actual': y,
        'prediction': preds
    })
    out_dir = os.path.join(DATA_DIR, 'final_output')
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'lstm_augmented_test_results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"Saved LSTM augmented test results to {results_path}")


def main():
    logger = setup_logger()
    logger.info("Starting LSTM augmented testing...")
    model, scaler = load_model_and_scaler()
    X, y, timestamps, transaction_keys = load_test_data()
    evaluate_and_save(model, scaler, X, y, timestamps, transaction_keys, logger)
    logger.info("LSTM augmented testing completed.")


if __name__ == '__main__':
    main()
