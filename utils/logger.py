import os
import logging

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()


def setup_logging(log_dir: str = 'logs'):
    os.makedirs(log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    root_logger.setLevel(level)
    # Remove existing handlers to avoid duplicate logs
    root_logger.handlers = []
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'root.log'))
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)
    # Stream (console) handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)


def get_logger(name: str, log_dir: str = 'logs') -> logging.Logger:
    logger = logging.getLogger(name)
    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        # File handler for this logger
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

