# Enhanced Logging System

This project now includes a comprehensive, structured logging system that provides detailed, component-specific logging with proper timestamps and separation of concerns.

## Features

- **Component-specific log files**: Separate log files for different parts of the system
- **Structured logging**: JSON format support for machine-readable logs
- **Rotating file handlers**: Automatic log rotation to prevent disk space issues
- **Proper timestamps**: ISO format timestamps with timezone information
- **Error tracking**: Dedicated error logging with stack traces
- **Performance monitoring**: Duration tracking for operations

## Log Files

The logging system creates the following log files in the `logs/` directory:

- **`app.log`**: General application logs
- **`db.log`**: Database operations, queries, connections
- **`cli.log`**: Command-line interface operations
- **`model.log`**: Model training, evaluation, hyperparameter optimization
- **`pipeline.log`**: Pipeline stages, data flow, processing steps
- **`data.log`**: Data ingestion, preprocessing, feature engineering
- **`error.log`**: Error events with daily rotation (30-day retention)

## Configuration

Logging is configured in `config/config.yaml`:

```yaml
logging:
  level: INFO
  json: true                    # Use JSON format for structured logs
  base_log_dir: logs           # Base directory for log files
  max_bytes: 10485760          # 10MB per log file before rotation
  backup_count: 5              # Keep 5 backup files
  # Component-specific logging levels (optional overrides)
  components:
    db: INFO
    cli: INFO
    model: INFO
    pipeline: INFO
    data: DEBUG
    error: ERROR
```

## Usage

### Basic Usage

```python
from app.logging import get_logger, get_db_logger, get_model_logger

# Get component-specific loggers
app_log = get_logger("app")
db_log = get_db_logger()
model_log = get_model_logger()

# Log structured data
app_log.info("operation_started", 
             operation="train_model",
             config_file="config.yaml",
             dry_run=False)

db_log.info("query_executed",
           query="SELECT * FROM predictions",
           rows_returned=1500,
           execution_time_ms=245)

model_log.info("hyperparameter_optimization_completed",
              model="lightgbm",
              best_score=0.85,
              trials_completed=100)
```

### Available Logger Functions

```python
from app.logging import (
    setup_logging,       # Initialize the logging system
    get_logger,          # Get general logger
    get_db_logger,       # Database operations
    get_cli_logger,      # CLI commands
    get_model_logger,    # Model training/prediction
    get_pipeline_logger, # Pipeline operations
    get_data_logger,     # Data processing
    get_error_logger     # Error logging
)
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational information
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error events but application continues
- **CRITICAL**: Critical errors that may cause application to terminate

### Structured Logging Best Practices

1. **Use descriptive event names**: `"model_training_started"` instead of `"Starting training"`
2. **Include relevant context**: Add operation IDs, timestamps, parameters
3. **Use consistent field names**: `duration_seconds`, `error_type`, `rows_processed`
4. **Log performance metrics**: Execution times, memory usage, throughput
5. **Include error context**: Stack traces, input parameters that caused errors

### Example Log Entries

```json
{
  "timestamp": "2025-10-13T10:30:45.123456",
  "level": "info",
  "event": "model_training_completed",
  "model": "lightgbm",
  "training_time_seconds": 245.67,
  "samples": 10000,
  "features": 45,
  "final_score": 0.8543
}

{
  "timestamp": "2025-10-13T10:32:15.789012", 
  "level": "error",
  "event": "database_connection_failed",
  "host": "prod-db-01",
  "database": "predictions",
  "error": "Connection timeout after 30 seconds",
  "error_type": "ConnectionTimeoutError",
  "retry_attempt": 3
}
```

## Testing

Run the logging test script to verify the system:

```bash
python test_logging.py
```

This will create sample log entries in all component log files and demonstrate the structured logging capabilities.

## Log Rotation

- **File size rotation**: When log files reach 10MB, they are rotated
- **Backup retention**: 5 backup files are kept per component
- **Error log rotation**: Daily rotation with 30-day retention
- **Automatic cleanup**: Old log files are automatically removed

## Monitoring and Analysis

The structured JSON logs can be easily ingested by log analysis tools like:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Splunk**
- **Grafana Loki**
- **CloudWatch Logs**
- **Fluentd/Fluent Bit**

Query examples:
```bash
# Find all model training events
grep "model_training" logs/model.log

# Find errors in the last hour
grep "$(date -d '1 hour ago' '+%Y-%m-%dT%H')" logs/error.log

# Parse JSON logs with jq
cat logs/app.log | jq 'select(.level == "error")'
```
