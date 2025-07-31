import os
import logging
import json
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, MODEL_DIR


def setup_logger():
    logger = logging.getLogger('06_train_mlp_augmented_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_mlp_augmented_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_augmented.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    X = df.drop(columns=['datetime', 'TokenCount'])
    y = df['TokenCount']
    return X, y


def load_hyperparams():
    model_dir = os.path.join(MODEL_DIR, 'mlp_augmented')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': 42
        }
    return params, model_dir


def save_model_and_params(model, scaler, params, model_dir, logger, feature_columns):
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved trained MLP (augmented) model to {model_path}")
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)  # Save the correct scaler
    logger.info(f"Saved scaler to {scaler_path}")
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {hp_path}")
    feat_path = os.path.join(model_dir, 'feature_columns.json')
    with open(feat_path, 'w') as f:
        json.dump(feature_columns, f)
    logger.info(f"Saved feature columns to {feat_path}")


def main():
    logger = setup_logger()
    logger.info("Starting MLP (augmented) training...")

    X, y = load_data()
    logger.info(f"Loaded training data X:{X.shape}, y:{y.shape}")

    # Fit scaler on DataFrame to preserve column names
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_columns = list(X.columns)

    params, model_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    model = MLPRegressor(**params)
    model.fit(X_scaled, y)
    logger.info("MLP (augmented) training completed.")

    save_model_and_params(model, scaler, params, model_dir, logger, feature_columns)
    logger.info("Training agent finished successfully.")


if __name__ == '__main__':
    main()