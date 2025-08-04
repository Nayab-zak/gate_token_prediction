# Agent for data splitting
import os
import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from config import DATA_DIR, TEST_SPLIT_MONTHS, VALIDATION_SPLIT_MONTHS


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
    
    # Calculate cutoff dates for test and validation sets
    test_cutoff = last_date - relativedelta(months=TEST_SPLIT_MONTHS)
    validation_cutoff = test_cutoff - relativedelta(months=VALIDATION_SPLIT_MONTHS)
    
    # Split into three sets: train, validation, and test
    train = df[df['datetime'] <= validation_cutoff].copy()
    validation = df[(df['datetime'] > validation_cutoff) & (df['datetime'] <= test_cutoff)].copy()
    test = df[df['datetime'] > test_cutoff].copy()

    # Ensure output dirs
    out_dir = os.path.join(DATA_DIR, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    # Save the three datasets
    train_path = os.path.join(out_dir, 'train.csv')
    validation_path = os.path.join(out_dir, 'validation.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    
    logger.info(f"Data split done. Train: {train.shape}, Validation: {validation.shape}, Test: {test.shape}")
    logger.info(f"Cutoff dates - Validation: {validation_cutoff}, Test: {test_cutoff}")
    
    # Verify temporal integrity
    logger.info(f"Train date range: {train['datetime'].min()} to {train['datetime'].max()}")
    logger.info(f"Validation date range: {validation['datetime'].min()} to {validation['datetime'].max()}")
    logger.info(f"Test date range: {test['datetime'].min()} to {test['datetime'].max()}")
    
    return train, validation, test


def main():
    logger = setup_logger()
    logger.info("Starting data splitting with temporal validation...")
    df = load_preprocessed()
    train, validation, test = split_data(df, logger)
    
    # Calculate and log split proportions
    total_rows = len(df)
    train_pct = len(train) / total_rows * 100
    val_pct = len(validation) / total_rows * 100
    test_pct = len(test) / total_rows * 100
    
    logger.info(f"Split proportions: Train={train_pct:.1f}%, Validation={val_pct:.1f}%, Test={test_pct:.1f}%")
    logger.info("Data splitting completed with train, validation, and test sets.")


if __name__ == '__main__':
    main()
