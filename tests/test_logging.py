#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced logging system
"""
from app.logging import setup_logging, get_logger, get_db_logger, get_cli_logger, get_model_logger, get_pipeline_logger, get_data_logger
import time

def test_logging_system():
    # Setup logging
    setup_logging(
        level="INFO",
        json_logs=False,  # Use readable format for testing
        base_log_dir="logs",
        max_bytes=1048576,
        backup_count=3
    )
    
    # Test different loggers
    app_log = get_logger("app")
    db_log = get_db_logger()
    cli_log = get_cli_logger()
    model_log = get_model_logger()
    pipeline_log = get_pipeline_logger()
    data_log = get_data_logger()
    
    # Test various log levels and structured data
    app_log.info("logging_system_test_started", test_id="test_001")
    
    # Simulate database operations
    db_log.info("database_connection_established", 
                host="test-host", 
                database="test_db",
                connection_pool_size=10)
    
    db_log.debug("executing_query", 
                 query="SELECT * FROM test_table",
                 parameters={"start_date": "2025-01-01", "limit": 100})
    
    # Simulate CLI operations
    cli_log.info("command_executed", 
                command="train",
                config_file="config/config.yaml",
                dry_run=False)
    
    # Simulate model training
    model_log.info("model_training_started",
                  model_type="lightgbm",
                  hyperparameters={"n_estimators": 100, "learning_rate": 0.1})
    
    model_log.info("fold_completed",
                  fold=1,
                  total_folds=5,
                  metrics={"mae": 1.23, "rmse": 2.45})
    
    # Simulate pipeline operations
    pipeline_log.info("pipeline_stage_started",
                     stage="feature_engineering",
                     input_rows=10000)
    
    time.sleep(1)  # Simulate processing time
    
    pipeline_log.info("pipeline_stage_completed",
                     stage="feature_engineering",
                     output_rows=9500,
                     duration_seconds=1.0)
    
    # Simulate data processing
    data_log.info("data_ingestion_started",
                 source="database",
                 table="raw_data")
    
    data_log.warning("data_quality_issue",
                    issue="missing_values",
                    affected_columns=["column_a", "column_b"],
                    missing_percentage=5.2)
    
    data_log.info("data_preprocessing_completed",
                 original_shape=(10000, 50),
                 processed_shape=(9500, 65),
                 new_features_added=15)
    
    # Test error logging
    try:
        # Simulate an error
        raise ValueError("This is a test error for logging demonstration")
    except Exception as e:
        app_log.error("test_error_occurred",
                     error=str(e),
                     error_type=type(e).__name__,
                     test_context="logging_demonstration")
    
    app_log.info("logging_system_test_completed", 
                test_id="test_001",
                success=True)
    
    print("\n" + "="*60)
    print("Logging test completed!")
    print("Check the following log files:")
    print("- logs/app.log")
    print("- logs/db.log") 
    print("- logs/cli.log")
    print("- logs/model.log")
    print("- logs/pipeline.log")
    print("- logs/data.log")
    print("- logs/error.log")
    print("="*60)

if __name__ == "__main__":
    test_logging_system()
