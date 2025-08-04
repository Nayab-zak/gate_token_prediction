import os
import logging
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from config import DATA_DIR, MODEL_DIR, RF_AUGMENTED_ESTIMATORS


def setup_logger():
    logger = logging.getLogger('06_train_rf_augmented_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/06_train_rf_augmented_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data():
    # Load augmented encoded training data
    path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_augmented.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    X = df.drop(columns=['datetime', 'TokenCount'])
    y = df['TokenCount']
    return X, y


def load_hyperparams():
    # Directory for augmented RF
    hp_dir = os.path.join(MODEL_DIR, 'rf_augmented')
    os.makedirs(hp_dir, exist_ok=True)
    hp_path = os.path.join(hp_dir, 'hyperparam.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            params = json.load(f)
    else:
        # default hyperparameters using estimators from config
        params = {"n_estimators": RF_AUGMENTED_ESTIMATORS, "max_depth": 10, "random_state": 42}
    return params, hp_dir


def save_model_and_params(model, params, hp_dir, logger):
    # Save trained model and hyperparameters
    model_path = os.path.join(hp_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved trained RandomForest (augmented) model to {model_path}")
    hp_path = os.path.join(hp_dir, 'hyperparam.json')
    with open(hp_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {hp_path}")


def main():
    logger = setup_logger()
    logger.info("Starting RandomForest (augmented) training...")

    X, y = load_data()
    logger.info(f"Loaded training data X:{X.shape}, y:{y.shape}")

    params, hp_dir = load_hyperparams()
    logger.info(f"Using hyperparameters: {params}")

    model = RandomForestRegressor(**params)
    model.fit(X, y)
    logger.info("RandomForest (augmented) training completed.")

    save_model_and_params(model, params, hp_dir, logger)
    logger.info("Training agent finished successfully.")


if __name__ == '__main__':
    main()
