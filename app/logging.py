from __future__ import annotations
import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import structlog
from datetime import datetime
from typing import Dict, Optional

class LoggerManager:
    """Centralized logger manager for different components"""
    
    def __init__(self, base_log_dir: str = "logs", level: str = "INFO", 
                 json_logs: bool = True, max_bytes: int = 10_485_760, 
                 backup_count: int = 5):
        self.base_log_dir = Path(base_log_dir)
        self.level = level
        self.json_logs = json_logs
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.loggers: Dict[str, structlog.stdlib.BoundLogger] = {}
        
        # Ensure log directory exists
        self.base_log_dir.mkdir(exist_ok=True)
        
        # Component-specific log files
        self.log_files = {
            "app": self.base_log_dir / "app.log",
            "db": self.base_log_dir / "db.log", 
            "cli": self.base_log_dir / "cli.log",
            "model": self.base_log_dir / "model.log",
            "pipeline": self.base_log_dir / "pipeline.log",
            "data": self.base_log_dir / "data.log",
            "error": self.base_log_dir / "error.log"
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration for all components"""
        log_level = getattr(logging, self.level.upper(), logging.INFO)
        
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Setup base formatter
        if self.json_logs:
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ]
        else:
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
            wrapper_class=structlog.stdlib.BoundLogger,
        )
        
        # Setup console handler for all loggers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # Create component-specific loggers
        for component, log_file in self.log_files.items():
            logger = logging.getLogger(component)
            logger.setLevel(log_level)
            logger.handlers.clear()
            
            # Add console handler
            logger.addHandler(console_handler)
            
            # Add file handler with rotation
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=self.max_bytes, 
                backupCount=self.backup_count
            )
            file_handler.setLevel(log_level)
            
            if self.json_logs:
                # For JSON logs, structlog will handle the formatting
                file_formatter = logging.Formatter('%(message)s')
            else:
                # For text logs, use detailed formatting with timestamps
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Store structlog bound logger
            self.loggers[component] = structlog.get_logger(component)
        
        # Setup error logger with daily rotation
        error_logger = logging.getLogger("error")
        error_handler = TimedRotatingFileHandler(
            self.log_files["error"],
            when="midnight",
            interval=1,
            backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        error_logger.addHandler(error_handler)
        
        # Log initialization
        self.get_logger("app").info(
            "logging_system_initialized",
            components=list(self.log_files.keys()),
            level=self.level,
            json_format=self.json_logs,
            log_directory=str(self.base_log_dir)
        )
    
    def get_logger(self, component: str = "app") -> structlog.stdlib.BoundLogger:
        """Get logger for specific component"""
        if component not in self.loggers:
            # Create new logger if not exists
            logger = logging.getLogger(component)
            self.loggers[component] = structlog.get_logger(component)
        
        return self.loggers[component]

# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None

def setup_logging(level: str = "INFO", json_logs: bool = True, 
                  base_log_dir: str = "logs", max_bytes: int = 10_485_760, 
                  backup_count: int = 5) -> LoggerManager:
    """Initialize the global logging system"""
    global _logger_manager
    _logger_manager = LoggerManager(
        base_log_dir=base_log_dir,
        level=level,
        json_logs=json_logs,
        max_bytes=max_bytes,
        backup_count=backup_count
    )
    return _logger_manager

def get_logger(component: str = "app") -> structlog.stdlib.BoundLogger:
    """Get logger for specific component"""
    if _logger_manager is None:
        setup_logging()  # Initialize with defaults
    return _logger_manager.get_logger(component)

# Convenience functions for different components
def get_db_logger() -> structlog.stdlib.BoundLogger:
    """Get database operations logger"""
    return get_logger("db")

def get_cli_logger() -> structlog.stdlib.BoundLogger:
    """Get CLI operations logger"""
    return get_logger("cli")

def get_model_logger() -> structlog.stdlib.BoundLogger:
    """Get model training/prediction logger"""
    return get_logger("model")

def get_pipeline_logger() -> structlog.stdlib.BoundLogger:
    """Get pipeline operations logger"""
    return get_logger("pipeline")

def get_data_logger() -> structlog.stdlib.BoundLogger:
    """Get data processing logger"""
    return get_logger("data")

def get_error_logger() -> structlog.stdlib.BoundLogger:
    """Get error logger"""
    return get_logger("error")
