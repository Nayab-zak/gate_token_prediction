#!/usr/bin/env python3
# filepath: /home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/tests/test_01_data_ingestion.py

import os
import sys
import logging
import pandas as pd
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR
from agents.01_data_ingestion_agent import load_input_data, setup_logger

# Set up test logger
def setup_test_logger():
    logger = logging.getLogger('test_01_data_ingestion')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/tests', exist_ok=True)
    fh = logging.FileHandler('logs/tests/test_01_data_ingestion.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def create_test_excel():
    """Create a test Excel file with the specific sheet name"""
    logger = setup_test_logger()
    logger.info("Creating test Excel file with 'Token_Input_data_desig' sheet")
    
    # Ensure test data directories exist
    raw_dir = os.path.join(DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Create test data
    test_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(10)],
        'gate_id': [f'G{i}' for i in range(10)],
        'token_count': [100 + i * 10 for i in range(10)],
        'is_holiday': [False for _ in range(10)],
        'temperature': [25 + i for i in range(10)],
    }
    df = pd.DataFrame(test_data)
    
    # Create Excel file with specific sheet name
    test_file_path = os.path.join(raw_dir, 'test_token_data.xlsx')
    with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Token_Input_data_desig', index=False)
        # Add another sheet for testing sheet detection
        df.to_excel(writer, sheet_name='other_sheet', index=False)
    
    logger.info(f"Test Excel file created at: {test_file_path}")
    return test_file_path

def create_test_csv():
    """Create a test CSV file"""
    logger = setup_test_logger()
    logger.info("Creating test CSV file")
    
    # Ensure test data directories exist
    raw_dir = os.path.join(DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Create test data
    test_data = {
        'datetime': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(10)],
        'GateID': [f'G{i}' for i in range(10)],
        'TokenCount': [100 + i * 10 for i in range(10)],
        'IsHoliday': [False for _ in range(10)],
        'Temperature': [25 + i for i in range(10)],
    }
    df = pd.DataFrame(test_data)
    
    # Create CSV file
    test_file_path = os.path.join(raw_dir, 'test_moves.csv')
    df.to_csv(test_file_path, index=False)
    
    logger.info(f"Test CSV file created at: {test_file_path}")
    return test_file_path

def test_data_ingestion():
    """Test both CSV and Excel ingestion functionality"""
    logger = setup_test_logger()
    logger.info("=== Starting data ingestion test ===")
    
    # Create test files
    excel_file_path = create_test_excel()
    csv_file_path = create_test_csv()
    
    # Import the data ingestion agent module
    import importlib
    ingestion_agent = importlib.import_module('agents.01_data_ingestion_agent')
    
    # Process the files
    logger.info("Testing data ingestion functions...")
    ingestion_agent.load_input_data(logger)
    
    # Check if CSV files were created in the preprocessed directory
    processed_dir = os.path.join(DATA_DIR, 'preprocessed')
    excel_output_csv = os.path.join(processed_dir, 'test_token_data_token_data.csv')
    csv_output = os.path.join(processed_dir, 'test_moves_processed.csv')
    
    success = True
    
    # Check Excel-derived file
    if os.path.exists(excel_output_csv):
        logger.info(f"✅ SUCCESS: Excel-derived CSV file correctly created at {excel_output_csv}")
        df = pd.read_csv(excel_output_csv)
        logger.info(f"Excel-derived CSV contains {len(df)} rows")
        logger.info(f"Columns: {', '.join(df.columns)}")
    else:
        logger.error(f"❌ FAILED: Excel-derived CSV file not created at {excel_output_csv}")
        success = False
    
    # Check direct CSV file
    if os.path.exists(csv_output):
        logger.info(f"✅ SUCCESS: Direct CSV file correctly processed at {csv_output}")
        df = pd.read_csv(csv_output)
        logger.info(f"Processed CSV contains {len(df)} rows")
        logger.info(f"Columns: {', '.join(df.columns)}")
    else:
        logger.error(f"❌ FAILED: Direct CSV file not processed at {csv_output}")
        success = False
        
    return success

def main():
    logger = setup_test_logger()
    logger.info("Starting data ingestion agent tests")
    
    success = test_data_ingestion()
    
    if success:
        logger.info("All tests passed successfully! ✅")
    else:
        logger.error("Some tests failed. ❌")
        sys.exit(1)

if __name__ == "__main__":
    main()
    main()
 