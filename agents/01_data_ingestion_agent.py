import os
import shutil
import logging
from dotenv import load_dotenv
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


def main():
    logger = setup_logger()
    logger.info("Starting data ingestion...")
    ingest_raw_data(logger)
    logger.info("Data ingestion completed.")


if __name__ == '__main__':
    main()
