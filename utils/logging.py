# ===============================
# File: utils/logging.py
# ===============================
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_to_file: bool = True, log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger that outputs to both console and file.
    
    Args:
        name: Logger name (usually __name__)
        log_to_file: Whether to also log to a file
        log_dir: Directory to store log files
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Extract agent name from module name (e.g., "agents.ingestion_agent" -> "ingestion_agent")
        if "." in name:
            agent_name = name.split(".")[-1]
        else:
            agent_name = name.replace("__main__", "orchestrator")
            
        log_file = log_path / f"{agent_name}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also log to a combined log file
        combined_handler = logging.FileHandler(log_path / "pipeline.log")
        combined_handler.setFormatter(formatter)
        logger.addHandler(combined_handler)
    
    return logger