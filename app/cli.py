from __future__ import annotations
import typer
import os
from app.pipelines.train import run_train
from app.pipelines.predict import run_predict
from app.pipelines.backfill_actuals import run_backfill, compute_prediction_accuracy
from app.utils.ddl import create_or_update_dev_table
from app.utils.env import load_config_env
from app.logging import setup_logging, get_cli_logger

app = typer.Typer(add_completion=False)

def init_logging(config_path: str):
    """Initialize logging from config"""
    cfg = load_config_env(config_path)
    logging_cfg = cfg["logging"]
    
    setup_logging(
        level=logging_cfg["level"],
        json_logs=logging_cfg["json"], 
        base_log_dir=logging_cfg["base_log_dir"],
        max_bytes=logging_cfg["max_bytes"],
        backup_count=logging_cfg["backup_count"]
    )
    return cfg

@app.command()
def train(config_path: str = "config/config.yaml", dry_run: bool = False):
    """Train machine learning models"""
    logger = get_cli_logger()
    logger.info("train_command_started", config_path=config_path, dry_run=dry_run)
    
    try:
        cfg = init_logging(config_path)
        run_train(cfg, dry_run=dry_run)
        logger.info("train_command_completed", success=True)
    except Exception as e:
        logger.error("train_command_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(1)

@app.command()
def predict(config_path: str = "config/config.yaml", dry_run: bool = False):
    """Generate predictions using trained models"""
    logger = get_cli_logger()
    logger.info("predict_command_started", config_path=config_path, dry_run=dry_run)
    
    try:
        cfg = init_logging(config_path)
        run_predict(cfg, dry_run=dry_run)
        logger.info("predict_command_completed", success=True)
    except Exception as e:
        logger.error("predict_command_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(1)

@app.command("create-dev-table")
def create_dev_table(table: str = None, schema: str = None, config_realtime: str = "config/realtime.yaml",
                     dry_run: bool = False, apply: bool = False):
    """Create or update development table schema"""
    logger = get_cli_logger()
    logger.info("create_dev_table_started", 
                table=table, schema=schema, dry_run=dry_run, apply=apply)
    
    try:
        cfg = init_logging("config/config.yaml")
        
        # Use schema from environment if not provided
        if schema is None:
            schema = os.getenv('DEFAULT_SCHEMA', 'DPW_DL')
        
        # Use table from environment if not provided
        if table is None:
            table = os.getenv('DEFAULT_TABLE', 'T_DA_PRED_GATE_TOKEN')
        
        logger.info("resolved_table_details", schema=schema, table=table)
        
        create_or_update_dev_table(schema, table, config_realtime, "", dry_run=dry_run, apply=apply)
        logger.info("create_dev_table_completed", success=True)
    except Exception as e:
        logger.error("create_dev_table_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(1)

@app.command()
def backfill(config_path: str = "config/config.yaml", dry_run: bool = False):
    """Backfill actual values for predictions when they become available."""
    logger = get_cli_logger()
    logger.info("backfill_command_started", config_path=config_path, dry_run=dry_run)
    
    try:
        cfg = init_logging(config_path)
        run_backfill(cfg, dry_run=dry_run)
        logger.info("backfill_command_completed", success=True)
    except Exception as e:
        logger.error("backfill_command_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(1)

@app.command("accuracy")
def accuracy(config_path: str = "config/config.yaml", days_back: int = 7):
    """Compute prediction accuracy metrics for predictions with actuals."""
    logger = get_cli_logger()
    logger.info("accuracy_command_started", config_path=config_path, days_back=days_back)
    
    try:
        cfg = init_logging(config_path)
        metrics = compute_prediction_accuracy(cfg, days_back=days_back)
        
        if metrics.height > 0:
            logger.info("accuracy_metrics_computed", 
                       metrics_count=metrics.height,
                       columns=metrics.columns)
            print("\nPrediction Accuracy Metrics:")
            print(metrics.to_pandas().to_string(index=False))
        else:
            logger.warning("no_accuracy_metrics_found", days_back=days_back)
            print("No prediction accuracy metrics found for the specified period.")
            
        logger.info("accuracy_command_completed", success=True)
    except Exception as e:
        logger.error("accuracy_command_failed", error=str(e), error_type=type(e).__name__)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
