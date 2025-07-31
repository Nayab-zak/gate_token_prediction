import os
import logging
from dotenv import load_dotenv
from config import DATA_DIR, MODEL_DIR, UAE_HOLIDAYS, TOP_IF_SLIDER, TEST_SPLIT_MONTHS, FORECAST_HORIZON


def setup_logger():
    logger = logging.getLogger('00_config_validation_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/00_config_validation_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def validate_env_vars(logger):
    logger.info("Loading environment variables from .env")
    load_dotenv()
    required = ['DATA_DIR', 'MODEL_DIR']
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        msg = f"Missing required environment variables: {missing}"
        logger.error(msg)
        raise EnvironmentError(msg)
    logger.info("Environment variables loaded successfully")


def validate_paths(logger):
    logger.info("Validating directory paths from config.py")
    # Ensure DATA_DIR and subdirectories
    subdirs = ['raw', 'preprocessed', 'features', 'encoded_input', 'encoded_output', 'final_output']
    for sub in subdirs:
        path = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(path):
            logger.warning(f"Directory not found, creating: {path}")
            os.makedirs(path, exist_ok=True)
        else:
            logger.info(f"Found directory: {path}")

    # Ensure MODEL_DIR exists
    if not os.path.isdir(MODEL_DIR):
        logger.warning(f"MODEL_DIR not found, creating: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
    else:
        logger.info(f"Found MODEL_DIR: {MODEL_DIR}")


def validate_parameters(logger):
    logger.info("Validating other config parameters")
    # UAE_HOLIDAYS
    if not isinstance(UAE_HOLIDAYS, list):
        msg = "UAE_HOLIDAYS must be a list of date strings"
        logger.error(msg)
        raise ValueError(msg)
    # TOP_IF_SLIDER
    if not isinstance(TOP_IF_SLIDER, list) or len(TOP_IF_SLIDER) == 0:
        msg = "TOP_IF_SLIDER must be a non-empty list of feature names"
        logger.error(msg)
        raise ValueError(msg)
    # TEST_SPLIT_MONTHS & FORECAST_HORIZON
    for param, name in [(TEST_SPLIT_MONTHS, 'TEST_SPLIT_MONTHS'), (FORECAST_HORIZON, 'FORECAST_HORIZON')]:
        if not isinstance(param, int) or param <= 0:
            msg = f"{name} must be a positive integer"
            logger.error(msg)
            raise ValueError(msg)
    logger.info("All config parameters validated successfully")


def main():
    logger = setup_logger()
    try:
        validate_env_vars(logger)
        validate_paths(logger)
        validate_parameters(logger)
        logger.info("Configuration validation completed without errors.")
    except Exception as e:
        logger.exception("Configuration validation failed")
        raise


if __name__ == '__main__':
    main()
