#!/usr/bin/env python3
# filepath: /home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/tests/test_03_splitting_agent.py

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR, TEST_SPLIT_MONTHS, VALIDATION_SPLIT_MONTHS

# Import the splitting agent modules - use importlib to handle numeric module names
import importlib.util
spec = importlib.util.spec_from_file_location("splitting_agent", 
                                             os.path.join(os.path.dirname(__file__), 
                                             "..", "agents", "03_splitting_agent.py"))
splitting_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitting_agent)

# Now we can access the functions
split_data = splitting_agent.split_data
setup_logger = splitting_agent.setup_logger

# Set up test logger
def setup_test_logger():
    logger = logging.getLogger('test_03_splitting_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/tests', exist_ok=True)
    fh = logging.FileHandler('logs/tests/test_03_splitting_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def create_test_data():
    """Create a test dataset with timestamps"""
    logger = setup_test_logger()
    logger.info("Creating test time series data")
    
    # Create dates for 24 months of hourly data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(hours=1)
    
    # Create dataframe with dummy data
    df = pd.DataFrame({
        'datetime': dates,
        'token_count': np.random.randint(50, 500, len(dates)),
        'feature1': np.random.random(len(dates)),
        'feature2': np.random.random(len(dates))
    })
    
    # Ensure test data directory exists
    os.makedirs(os.path.join(DATA_DIR, 'preprocessed'), exist_ok=True)
    
    # Save as preprocessed.csv for testing
    preprocessed_path = os.path.join(DATA_DIR, 'preprocessed', 'test_preprocessed.csv')
    df.to_csv(preprocessed_path, index=False)
    
    logger.info(f"Created test dataset with {len(df)} rows from {start_date} to {end_date}")
    return df, preprocessed_path

def test_time_based_split():
    """Test the time-based split with validation set"""
    logger = setup_test_logger()
    logger.info("=== Starting time-based split test ===")
    
    # Create test data
    df, _ = create_test_data()
    
    # Perform split
    logger.info(f"Splitting with TEST_SPLIT_MONTHS={TEST_SPLIT_MONTHS}, VALIDATION_SPLIT_MONTHS={VALIDATION_SPLIT_MONTHS}")
    train, validation, test = split_data(df, logger)
    
    # Verify splits
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"Train set: {len(train)} rows ({len(train)/total_rows*100:.1f}%)")
    logger.info(f"Validation set: {len(validation)} rows ({len(validation)/total_rows*100:.1f}%)")
    logger.info(f"Test set: {len(test)} rows ({len(test)/total_rows*100:.1f}%)")
    
    # Check temporal integrity
    assert train['datetime'].max() < validation['datetime'].min(), "Train/validation overlap detected!"
    assert validation['datetime'].max() < test['datetime'].min(), "Validation/test overlap detected!"
    
    # Check for expected proportion (rough check)
    test_months_pct = TEST_SPLIT_MONTHS / 24 * 100  # Assuming 24 months of data
    val_months_pct = VALIDATION_SPLIT_MONTHS / 24 * 100
    
    test_pct = len(test) / total_rows * 100
    val_pct = len(validation) / total_rows * 100
    
    logger.info(f"Test split: Expected ~{test_months_pct:.1f}%, Got {test_pct:.1f}%")
    logger.info(f"Validation split: Expected ~{val_months_pct:.1f}%, Got {val_pct:.1f}%")
    
    # Check for data files
    train_path = os.path.join(DATA_DIR, 'preprocessed', 'train.csv')
    val_path = os.path.join(DATA_DIR, 'preprocessed', 'validation.csv')
    test_path = os.path.join(DATA_DIR, 'preprocessed', 'test.csv')
    
    assert os.path.exists(train_path), f"Train file not created: {train_path}"
    assert os.path.exists(val_path), f"Validation file not created: {val_path}"
    assert os.path.exists(test_path), f"Test file not created: {test_path}"
    
    logger.info("✅ Time-based split test passed!")
    return True

def main():
    logger = setup_test_logger()
    logger.info("Starting data splitting agent tests")
    
    success = test_time_based_split()
    
    if success:
        logger.info("All tests passed successfully! ✅")
    else:
        logger.error("Some tests failed. ❌")
        sys.exit(1)

if __name__ == "__main__":
    main()
