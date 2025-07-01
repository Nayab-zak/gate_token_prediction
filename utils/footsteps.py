import os
import logging
import functools
from datetime import datetime

LOG_PATH = os.path.join('logs', 'footsteps.log')
os.makedirs('logs', exist_ok=True)

# Configure footsteps logger
foot_logger = logging.getLogger('footsteps')
if not foot_logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fmt = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fmt)
    foot_logger.addHandler(fh)
foot_logger.propagate = False


def track(step_name: str):
    """Log a timestamped step to footsteps.log."""
    foot_logger.info(f"STEP: {step_name}")


def track_step(step_name: str):
    """Decorator to log entry and exit of a function/method."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            foot_logger.info(f"ENTER: {step_name}")
            result = func(*args, **kwargs)
            foot_logger.info(f"EXIT: {step_name}")
            return result
        return wrapper
    return decorator
