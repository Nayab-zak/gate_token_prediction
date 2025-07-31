import os
import logging
import json
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from config import DATA_DIR, MODEL_DIR

# Note: This agent assumes the preceding agents have been run to generate
# raw data, preprocess, feature engineering, and encoding for real-time inputs.


def setup_logger():
    logger = logging.getLogger('10_real_time_prediction_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/10_real_time_prediction_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_champion_info(logger):
    champ_path = os.path.join(MODEL_DIR, 'champion_model', 'champ.json')
    if not os.path.exists(champ_path):
        raise FileNotFoundError(f"Champion info not found: {champ_path}")
    with open(champ_path, 'r') as f:
        data = json.load(f)
    champion = data.get('champion')
    logger.info(f"Champion model: {champion}")
    return champion


def load_real_time_input(pipeline_type):
    # Determine file name based on pipeline (classic vs augmented)
    input_file = os.path.join(DATA_DIR, 'encoded_input', f'realtime_input_{pipeline_type}.csv')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Real-time input file not found: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime'])
    timestamps = df['datetime']
    X = df.drop(columns=['datetime', 'TokenCount'])
    return X, timestamps


def load_pipeline_models(champion):
    # champion format: '<model>_<pipeline>' e.g. 'xgb_augmented'
    model_name, pipeline = champion.split('_', 1)
    model_dir = os.path.join(MODEL_DIR, champion)

    # Load scaler if exists
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Load model
    if model_name in ['lstm']:
        model_path = os.path.join(model_dir, 'trained_model.h5')
        model = load_model(model_path)
    else:
        model_path = os.path.join(model_dir, 'trained_model.pkl')
        model = joblib.load(model_path)

    return model, scaler, pipeline


def predict(model, scaler, pipeline, X):
    # Apply scaler if provided
    if scaler:
        X_proc = scaler.transform(X)
    else:
        X_proc = X.values if hasattr(X, 'values') else X

    # Reshape for LSTM
    if pipeline.startswith('lstm') or 'lstm' in model.__class__.__name__.lower():
        X_proc = X_proc.reshape((X_proc.shape[0], 1, X_proc.shape[1]))

    preds = model.predict(X_proc)
    # Flatten LSTM outputs
    return preds.flatten() if hasattr(preds, 'flatten') else preds


def main():
    logger = setup_logger()
    logger.info("Starting real-time prediction agent...")

    # Load champion model info
    champion = load_champion_info(logger)
    model, scaler, pipeline = load_pipeline_models(champion)

    # Load real-time input (classic or augmented) based on pipeline
    X, timestamps = load_real_time_input(pipeline)
    logger.info(f"Loaded real-time data: {X.shape[0]} rows")

    # Generate predictions
    preds = predict(model, scaler, pipeline, X)
    logger.info(f"Generated {len(preds)} real-time predictions.")

    # Save predictions
    results = pd.DataFrame({'datetime': timestamps, 'prediction': preds})
    out_dir = os.path.join(DATA_DIR, 'final_output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'real_time_predictions.csv')
    results.to_csv(out_path, index=False)
    logger.info(f"Saved real-time predictions to {out_path}")

    logger.info("Real-time prediction agent completed.")


if __name__ == '__main__':
    main()
