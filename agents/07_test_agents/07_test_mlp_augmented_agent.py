import os
import logging
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from config import DATA_DIR
import json


def setup_logger():
    logger = logging.getLogger('07_test_mlp_augmented_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/07_test_mlp_augmented_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_model_and_scaler():
    model_dir = os.getenv('MODEL_DIR', 'models')
    scaler_path = os.path.join(model_dir, 'mlp_augmented', 'scaler.pkl')
    model_path = os.path.join(model_dir, 'mlp_augmented', 'trained_model.pkl')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found in mlp_augmented directory")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


def load_feature_columns():
    model_dir = os.getenv('MODEL_DIR', 'models')
    feat_path = os.path.join(model_dir, 'mlp_augmented', 'feature_columns.json')
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature columns file not found: {feat_path}")
    with open(feat_path, 'r') as f:
        feature_columns = json.load(f)
    return feature_columns


def load_test_data():
    # Load encoded data for model prediction
    encoded_path = os.path.join(DATA_DIR, 'encoded_input', 'test_input_augmented.csv')
    df_encoded = pd.read_csv(encoded_path, parse_dates=['datetime'])
    feature_columns = load_feature_columns()
    X = df_encoded.drop(columns=['datetime', 'TokenCount'])
    # Align test features to training columns
    X = X.reindex(columns=feature_columns, fill_value=0)
    y = df_encoded['TokenCount']
    timestamps = df_encoded['datetime']
    
    # Load original features to get transaction-level identifiers
    features_path = os.path.join(DATA_DIR, 'features', 'test_features.csv')
    df_features = pd.read_csv(features_path, parse_dates=['datetime'])
    
    # Extract transaction identifiers (ensure same order as encoded data)
    df_features = df_features.sort_values('datetime').reset_index(drop=True)
    transaction_keys = df_features[['MoveType', 'TerminalID', 'Desig']].copy()
    
    return X, y, timestamps, transaction_keys


def evaluate_and_save(scaler, model, X, y, timestamps, transaction_keys, logger):
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    mape = np.mean(np.abs((y - preds) / y)) * 100
    logger.info(f"MLP Augmented Test - RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

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
    results_path = os.path.join(out_dir, 'mlp_augmented_test_results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"Saved MLP augmented test results to {results_path}")


def main():
    logger = setup_logger()
    logger.info("Starting MLP augmented testing...")
    scaler, model = load_model_and_scaler()
    X, y, timestamps, transaction_keys = load_test_data()
    evaluate_and_save(scaler, model, X, y, timestamps, transaction_keys, logger)
    logger.info("MLP augmented testing completed.")


if __name__ == '__main__':
    main()
