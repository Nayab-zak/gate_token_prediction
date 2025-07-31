import os
import logging
import json
import pandas as pd
import numpy as np
from config import DATA_DIR, MODEL_DIR


def setup_logger():
    logger = logging.getLogger('08_champion_selection_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/08_champion_selection_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_test_metrics(logger):
    results_dir = os.path.join(DATA_DIR, 'final_output')
    metrics = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith('_test_results.csv'):
            continue
        model_key = fname.replace('_test_results.csv', '')
        path = os.path.join(results_dir, fname)
        try:
            df = pd.read_csv(path)
            actual = df['actual'].values
            pred = df['prediction'].values
            rmse = np.sqrt(((pred - actual) ** 2).mean())
            mae = np.abs(pred - actual).mean()
            mape = (np.abs((pred - actual) / actual).mean()) * 100
            metrics[model_key] = {'rmse': float(rmse), 'mae': float(mae), 'mape': float(mape)}
            logger.info(f"Loaded metrics for {model_key}: RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%")
        except Exception as e:
            logger.error(f"Error loading {fname}: {e}")
    return metrics


def select_champion(metrics, logger):
    # Primary: lowest RMSE, Secondary: lowest MAE
    champion = None
    best_vals = (float('inf'), float('inf'))
    for model_key, m in metrics.items():
        vals = (m['rmse'], m['mae'])
        if vals < best_vals:
            best_vals = vals
            champion = model_key
    logger.info(f"Selected champion model: {champion} with RMSE={best_vals[0]:.3f}, MAE={best_vals[1]:.3f}")
    return champion, metrics[champion]


def save_champion(champion, stats, logger):
    champ_dir = os.path.join(MODEL_DIR, 'champion_model')
    os.makedirs(champ_dir, exist_ok=True)
    champ_path = os.path.join(champ_dir, 'champ.json')
    data = {'champion': champion, **stats}
    with open(champ_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved champion info to {champ_path}")


def main():
    logger = setup_logger()
    logger.info("Starting champion selection...")
    metrics = load_test_metrics(logger)
    if not metrics:
        logger.error("No test result files found to select champion.")
        return
    champion, stats = select_champion(metrics, logger)
    save_champion(champion, stats, logger)
    logger.info("Champion selection completed.")


if __name__ == '__main__':
    main()
