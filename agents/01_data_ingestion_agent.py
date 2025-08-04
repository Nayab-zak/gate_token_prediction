import os
import shutil
import logging
import pandas as pd
from dotenv import load_dotenv
import sys

# Add project root to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR


def setup_logger():
    logger = logging.getLogger('01_data_ingestion_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/01_data_ingestion_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def ingest_raw_data(logger):
    # Load environment vars
    load_dotenv()
    # Define source and destination directories
    src_dir = os.getenv('INGESTION_SOURCE_DIR', os.path.join(DATA_DIR, 'incoming'))
    raw_dir = os.path.join(DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        logger.warning(f"Source directory not found: {src_dir}. Ensure raw data files are placed there.")
        return

    # Copy new files
    ingested = []
    for fname in os.listdir(src_dir):
        if fname.lower().endswith(('.csv', '.xlsx')):
            src_path = os.path.join(src_dir, fname)
            dest_path = os.path.join(raw_dir, fname)
            # Avoid overwriting existing files
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                logger.info(f"Ingested file: {fname} -> {dest_path}")
                ingested.append(fname)
            else:
                logger.info(f"File already exists, skipping: {fname}")

    if not ingested:
        logger.info("No new raw data files ingested.")
    else:
        logger.info(f"Total files ingested: {len(ingested)}")


def load_input_data(logger):
    """
    Load data from both CSV files and Excel files:
    - For CSV files: Load directly
    - For Excel files: Look for sheet name 'Token_Input_data_desig'
    Save all extracted data as CSV for further processing.
    """
    raw_dir = os.path.join(DATA_DIR, 'raw')
    processed_dir = os.path.join(DATA_DIR, 'preprocessed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process Excel files with specific sheet
    excel_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.xlsx')]
    csv_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.csv')]
    
    if not excel_files and not csv_files:
        logger.warning("No input files (CSV or Excel) found in raw data directory.")
        return
    
    processed_count = 0
    sheet_name = 'Token_Input_data_desig'  # Specific sheet name to extract
    
    # Process Excel files - look for specific sheet
    for excel_file in excel_files:
        file_path = os.path.join(raw_dir, excel_file)
        try:
            # Check if the Excel file contains the specific sheet
            xls = pd.ExcelFile(file_path)
            if sheet_name in xls.sheet_names:
                # Load data from the specific sheet
                logger.info(f"Found target sheet '{sheet_name}' in {excel_file}, loading data...")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Save to CSV in the preprocessed folder
                output_filename = f"{os.path.splitext(excel_file)[0]}_token_data.csv"
                output_path = os.path.join(processed_dir, output_filename)
                
                # Verify if data was loaded correctly
                if df.empty:
                    logger.warning(f"Sheet '{sheet_name}' in {excel_file} is empty")
                else:
                    logger.info(f"Loaded {len(df)} rows from sheet '{sheet_name}'")
                    df.to_csv(output_path, index=False)
                    logger.info(f"Successfully extracted '{sheet_name}' sheet from {excel_file} to {output_filename}")
                    processed_count += 1
            else:
                available_sheets = ', '.join(xls.sheet_names)
                logger.warning(f"Excel file {excel_file} does not contain sheet '{sheet_name}'. Available sheets: {available_sheets}")
        except Exception as e:
            logger.error(f"Error processing Excel file {excel_file}: {str(e)}")
    
    # Process CSV files - load directly
    for csv_file in csv_files:
        file_path = os.path.join(raw_dir, csv_file)
        try:
            logger.info(f"Loading CSV file: {csv_file}")
            df = pd.read_csv(file_path)
            
            # Save to preprocessed folder with a consistent naming pattern
            output_filename = f"{os.path.splitext(csv_file)[0]}_processed.csv"
            output_path = os.path.join(processed_dir, output_filename)
            
            # Verify if data was loaded correctly
            if df.empty:
                logger.warning(f"CSV file {csv_file} is empty")
            else:
                logger.info(f"Loaded {len(df)} rows from {csv_file}")
                df.to_csv(output_path, index=False)
                logger.info(f"Successfully processed CSV file {csv_file} to {output_filename}")
                processed_count += 1
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_file}: {str(e)}")
    
    logger.info(f"Total processed files: {processed_count} (Excel with '{sheet_name}' sheet and CSV files)")


def main():
    logger = setup_logger()
    logger.info("Starting data ingestion...")
    ingest_raw_data(logger)
    logger.info("Loading data from input files (CSV directly, Excel with 'Token_Input_data_desig' sheet)...")
    load_input_data(logger)
    logger.info("Data ingestion completed.")


if __name__ == '__main__':
    main()
