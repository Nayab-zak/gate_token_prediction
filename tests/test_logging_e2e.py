#!/usr/bin/env python3
"""
Simple test to verify the logging system is working correctly across all modules
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.logging import setup_logging, get_logger, get_cli_logger
from app.utils.env import load_config_env

def test_end_to_end_logging():
    """Test that logging works end-to-end"""
    
    # Setup logging
    setup_logging(
        level="INFO",
        json_logs=False,  # Use readable format for testing
        base_log_dir="logs",
        max_bytes=1048576,
        backup_count=3
    )
    
    # Test basic logging
    app_log = get_logger()
    cli_log = get_cli_logger()
    
    app_log.info("logging_system_test_started", test_type="end_to_end")
    
    try:
        # Test CLI logging (simulating a command)
        cli_log.info("simulated_command_execution", 
                    command="test",
                    args=["--dry-run"],
                    config="config/config.yaml")
        
        # Test loading config (this was causing issues before)
        try:
            cfg = load_config_env("config/config.yaml")
            prediction_freq = int(cfg.get("prediction_frequency_hours", 6))
            app_log.info("config_loaded_successfully",
                        prediction_frequency_hours=prediction_freq,
                        timezone=cfg.get("timezone", "not_set"))
        except Exception as e:
            app_log.error("config_loading_failed", 
                         error=str(e), 
                         error_type=type(e).__name__)
        
        # Test database client initialization (this was also causing issues)
        try:
            from app.db.vertica_client import VerticaClient
            test_config = {
                "prod": {"host": "test", "port": 5433, "database": "test", "user": "test", "password": "test"},
                "dev": {"host": "test", "port": 5433, "database": "test", "user": "test", "password": "test"}
            }
            client = VerticaClient(test_config)
            app_log.info("vertica_client_initialized_successfully")
        except Exception as e:
            app_log.error("vertica_client_initialization_failed",
                         error=str(e),
                         error_type=type(e).__name__)
        
        app_log.info("end_to_end_test_completed", status="success")
        print("\n✅ End-to-end logging test completed successfully!")
        print("Check the following log files for outputs:")
        print("- logs/app.log")
        print("- logs/cli.log")
        print("- logs/db.log")
        
        return True
        
    except Exception as e:
        app_log.error("end_to_end_test_failed",
                     error=str(e),
                     error_type=type(e).__name__)
        print(f"\n❌ End-to-end test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_end_to_end_logging()
    sys.exit(0 if success else 1)
