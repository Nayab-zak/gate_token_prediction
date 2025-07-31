import os
import logging
import json
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from config import DATA_DIR, MODEL_DIR


def setup_logger():
    logger = logging.getLogger('06_train_lgbm_augmented_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_lgbm_augmented_agent.log')
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
    model_dir = os.path.join(MODEL_DIR, 'lgbm_augmented')
    os.makedirs(model_dir, exist_ok=True)
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42
        }
    return params, model_dir


def save_model_and_params(model, params, model_dir, logger):
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved trained LightGBM (augmented) model to {model_path}")
    hp_path = os.path.join(model_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting LightGBM (augmented) training...")

    X, y = load_data()
    logger.info(f"Loaded training data X:{X.shape}, y:{y.shape}")

    params, model_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    model = LGBMRegressor(**params)
    model.fit(X, y)
    logger.info("LightGBM (augmented) training completed.")

    save_model_and_params(model, params, model_dir, logger)
    logger.info("Training agent finished successfully.")


if __name__ == '__main__':
    main()