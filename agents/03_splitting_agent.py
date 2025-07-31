# Agent for data splitting
import os
import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from config import DATA_DIR, TEST_SPLIT_MONTHS


def setup_logger():
    logger = logging.getLogger('03_splitting_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/03_splitting_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_preprocessed():
    in_path = os.path.join(DATA_DIR, 'preprocessed', 'preprocessed.csv')
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Preprocessed file not found: {in_path}")
    return pd.read_csv(in_path, parse_dates=['datetime'])


def split_data(df, logger):
    df = df.sort_values('datetime')
    last_date = df['datetime'].max()
    cutoff = last_date - relativedelta(months=TEST_SPLIT_MONTHS)
    # Split
    train = df[df['datetime'] <= cutoff].copy()
    test = df[df['datetime'] > cutoff].copy()

    # Ensure output dirs
    out_dir = os.path.join(DATA_DIR, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, 'train.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info(f"Data split done. Train: {train.shape}, Test: {test.shape}, cutoff: {cutoff}")
    return train, test


def main():
    logger = setup_logger()
    logger.info("Starting data splitting...")
    df = load_preprocessed()
    split_data(df, logger)
    logger.info("Data splitting completed.")


if __name__ == '__main__':
    main()
