"""
Backfill actual values for predictions when they become available.
Runs hourly, same as the prediction pipeline.
"""
from __future__ import annotations
import datetime as dt
import polars as pl
import yaml
import time
import json
import os
from typing import Dict, Any
from app.db.vertica_client import VerticaClient
from app.utils.env import load_config_env
from app.logging import get_pipeline_logger, get_db_logger, get_data_logger, get_error_logger

def backfill_actuals(cfg: Dict[str, Any], dry_run: bool = False):
    """
    Backfill actual values for predictions using configurable prediction frequency.
    Runs at the same frequency as predictions to update with actual values when available.
    """
    pipeline_log = get_pipeline_logger()
    db_log = get_db_logger()
    data_log = get_data_logger()
    error_log = get_error_logger()
    
    t0 = time.time()
    
    pipeline_log.info("backfill_pipeline_started", dry_run=dry_run)
    
    try:
        dbconf = load_config_env("config/db.yaml")
        client = VerticaClient(dbconf)
        
        # Use same time calculation as predict pipeline
        end = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        # Get prediction frequency from config
        prediction_freq = int(cfg["prediction_frequency_hours"])
        
        # For backfill, look for predictions that were made for time periods
        # that have now passed and should have actual data available
        # Look back prediction_freq hours to find predictions that need actuals
        target_end = end - dt.timedelta(hours=prediction_freq)
        target_start = end - dt.timedelta(hours=prediction_freq * 2)
        
        pipeline_log.info("backfill_time_window_configured",
                         target_start=target_start.isoformat(),
                         target_end=target_end.isoformat(),
                         prediction_frequency_hours=prediction_freq,
                         current_time=end.isoformat())
    
        # Get actual data for the time period we're looking to backfill
        sql = open("db/sql/predict_query.sql", "r").read()
        
        data_log.info("fetching_actual_data_for_backfill",
                     time_window=f"{target_start.isoformat()} to {target_end.isoformat()}")
        
        try:
            # Get actual data for the target time period
            actual_rows = client.fetch_df("prod", sql, {
                "predict_start_ts": target_start, 
                "predict_end_ts": target_end
            })
            
            if actual_rows.height == 0:
                data_log.info("no_actual_data_available_for_backfill", 
                            target_start=target_start.isoformat(),
                            target_end=target_end.isoformat())
                return
            
            data_log.info("actual_data_fetched", 
                         rows=actual_rows.height,
                         time_window=f"{target_start.isoformat()} to {target_end.isoformat()}")
                
        except Exception as e:
            error_log.error("failed_to_fetch_actual_data", 
                           error=str(e), 
                           error_type=type(e).__name__)
            return
    
        # Find predictions that were made FOR the time period we now have actuals for
        # These are predictions that predicted the target_start to target_end window
        pred_table = f"{cfg['realtime']['dev_schema']}.{cfg['realtime']['dev_table']}"
        
        db_log.info("searching_for_predictions_to_backfill",
                   table=pred_table,
                   target_date_range=f"{target_start.date()} to {target_end.date()}")
        
        backfill_query = f"""
        SELECT 
            MoveDate,
            MoveHour,
            MoveType,
            TerminalID,
            Desig,
            TokenCount_pred,
            prediction_ts_utc,
            BI_BATCH_ID
        FROM {pred_table}
        WHERE TokenCount_actual IS NULL
          AND MoveDate >= :target_start_date
          AND MoveDate < :target_end_date
        """
        
        params = {
            "target_start_date": target_start.date(),
            "target_end_date": target_end.date()
        }
        
        try:
            predictions_to_update = client.fetch_df("dev", backfill_query, params)
            
            if predictions_to_update.height == 0:
                data_log.info("no_predictions_to_backfill",
                            time_window=f"{target_start.isoformat()} to {target_end.isoformat()}",
                            pred_table=pred_table)
                return
            
            data_log.info("predictions_found_for_backfill",
                         predictions_count=predictions_to_update.height,
                         time_window=f"{target_start.isoformat()} to {target_end.isoformat()}")
                
        except Exception as e:
            error_log.error("failed_to_fetch_predictions_for_backfill", 
                           error=str(e), 
                           error_type=type(e).__name__)
            return
    
        # Join predictions with actual data
        data_log.info("joining_predictions_with_actuals")
        actual_df = actual_rows.rename({"TokenCount": "TokenCount_actual"})
        
        # Convert to same data types for joining - handle the date type properly
        predictions_df = predictions_to_update.with_columns([
            pl.col("MoveHour").cast(pl.Int32)
        ])
        
        # The MoveDate from database is already a date, so we don't need to convert it
        # Just ensure both have the same type
        actual_df = actual_df.with_columns([
            pl.col("MoveHour").cast(pl.Int32)
        ])
        
        # Join on the key columns
        joined = predictions_df.join(
            actual_df,
            on=["MoveDate", "MoveHour", "MoveType", "TerminalID", "Desig"],
            how="inner"
        )
        
        if joined.height == 0:
            data_log.warning("no_matching_actuals_found",
                           predictions_checked=predictions_to_update.height,
                           actuals_available=actual_rows.height,
                           pred_sample=predictions_to_update.head(3).select(["MoveDate", "MoveHour", "MoveType", "TerminalID", "Desig"]).to_dict() if predictions_to_update.height > 0 else {},
                           actual_sample=actual_rows.head(3).select(["MoveDate", "MoveHour", "MoveType", "TerminalID", "Desig"]).to_dict() if actual_rows.height > 0 else {},
                           time_window_actual=f"{target_start.isoformat()} to {target_end.isoformat()}")
            return
        
        data_log.info("actuals_matched_with_predictions", 
                     matched_records=joined.height,
                     total_predictions_checked=predictions_to_update.height)
    
        if dry_run:
            pipeline_log.info("dry_run_backfill_preview", 
                             sample=joined.head(3).select([
                                 "MoveDate", "MoveHour", "MoveType", "TerminalID", "Desig",
                                 "TokenCount_pred", "TokenCount_actual", "BI_BATCH_ID"
                             ]).to_dict(),
                             total_rows=joined.height)
            return
        
        # Update predictions with actual values and calculate real production MAE
        db_log.info("starting_database_updates", records_to_update=joined.height)
        update_count = 0
        update_timestamp = dt.datetime.utcnow()
        
        for row_idx, row in enumerate(joined.iter_rows(named=True)):
            # Calculate individual prediction error (absolute error) - this is the real production MAE
            actual_val = float(row["TokenCount_actual"])
            pred_val = float(row["TokenCount_pred"])
            production_mae = abs(actual_val - pred_val)
            
            update_sql = f"""
            UPDATE {pred_table}
            SET TokenCount_actual = %s,
                primary_metric_value = %s,
                BI_UPDATED = %s
            WHERE MoveDate = %s 
              AND MoveHour = %s
              AND MoveType = %s
              AND TerminalID = %s
              AND Desig = %s
              AND BI_BATCH_ID = %s
            """
            
            update_params = [
                actual_val,
                round(production_mae, 2),
                update_timestamp,
                row["MoveDate"],
                int(row["MoveHour"]),
                str(row["MoveType"]),
                str(row["TerminalID"]),
                str(row["Desig"]),
                str(row["BI_BATCH_ID"])
            ]
            
            try:
                client.execute_dev_sql(update_sql, update_params)
                update_count += 1
                
                if (row_idx + 1) % 100 == 0:  # Log progress every 100 updates
                    db_log.debug("update_progress",
                               completed=row_idx + 1,
                               total=joined.height,
                               progress_percent=round((row_idx + 1) / joined.height * 100, 1))
                    
            except Exception as e:
                error_log.error("individual_update_failed", 
                               batch_id=row["BI_BATCH_ID"],
                               row_index=row_idx,
                               error=str(e),
                               error_type=type(e).__name__)
    
        runtime_sec = round(time.time() - t0, 3)
        
        # Save run metadata (same pattern as predict pipeline)
        run_meta = {
            "stage": "backfill",
            "rows_updated": update_count,
            "rows_checked": predictions_to_update.height,
            "duration_sec": runtime_sec,
            "time_window": {
                "target_start": target_start.isoformat(),
                "target_end": target_end.isoformat()
            },
            "prediction_frequency_hours": prediction_freq
        }
        
        os.makedirs(cfg["artifact_dir"], exist_ok=True)
        meta_filename = f"backfill_run_meta_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        run_meta_file = os.path.join(cfg["artifact_dir"], meta_filename)
        with open(run_meta_file, "w") as f:
            json.dump(run_meta, f, indent=2)
        
        pipeline_log.info("backfill_pipeline_completed", 
                         updated_rows=update_count,
                         checked_rows=predictions_to_update.height,
                         runtime_seconds=runtime_sec,
                         run_meta_file=run_meta_file)

    except Exception as e:
        error_log.error("backfill_pipeline_failed",
                       error=str(e),
                       error_type=type(e).__name__,
                       total_duration_seconds=round(time.time() - t0, 3))
        raise

def compute_prediction_accuracy(cfg: Dict[str, Any], days_back: int = 7):
    """
    Compute accuracy metrics for predictions that now have actuals.
    Uses real production MAE stored in model_cv_mae column after backfill.
    """
    pipeline_log = get_pipeline_logger()
    db_log = get_db_logger()
    error_log = get_error_logger()
    
    pipeline_log.info("computing_prediction_accuracy", days_back=days_back)
    
    try:
        dbconf = load_config_env("config/db.yaml")
        client = VerticaClient(dbconf)
        
        # Use same approach as predict pipeline for time windows
        end = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start = end - dt.timedelta(days=days_back)
        
        pred_table = f"{cfg['realtime']['dev_schema']}.{cfg['realtime']['dev_table']}"
        
        db_log.info("querying_accuracy_metrics",
                   table=pred_table,
                   time_window=f"{start.isoformat()} to {end.isoformat()}",
                   days_analyzed=days_back)
        
        accuracy_query = f"""
        SELECT 
            model_name,
            model_version,
            COUNT(*) as prediction_count,
            ROUND(AVG(primary_metric_value), 2) as avg_production_mae,
            ROUND(AVG(ABS(TokenCount_actual - TokenCount_pred)), 2) as calculated_mae,
            ROUND(SQRT(AVG(POWER(TokenCount_actual - TokenCount_pred, 2))), 2) as rmse,
            ROUND(AVG(TokenCount_actual), 2) as mean_actual,
            ROUND(AVG(TokenCount_pred), 2) as mean_predicted
        FROM {pred_table}
        WHERE TokenCount_actual IS NOT NULL
          AND primary_metric_value IS NOT NULL
          AND prediction_ts_utc >= :start_time
          AND prediction_ts_utc <= :end_time
        GROUP BY model_name, model_version
        ORDER BY model_version DESC
        """
        
        metrics = client.fetch_df("dev", accuracy_query, {
            "start_time": start,
            "end_time": end
        })
        
        if metrics.height > 0:
            pipeline_log.info("production_accuracy_metrics_computed", 
                             metrics_found=metrics.height,
                             metrics_summary=metrics.to_dict(),
                             days_analyzed=days_back)
        else:
            pipeline_log.warning("no_predictions_with_production_accuracy", 
                                days_back=days_back,
                                time_window=f"{start.isoformat()} to {end.isoformat()}")
                
        return metrics
        
    except Exception as e:
        error_log.error("accuracy_computation_failed", 
                       error=str(e), 
                       error_type=type(e).__name__,
                       days_back=days_back)
        raise

def run_backfill(cfg: Dict[str, Any], dry_run: bool = False):
    """Main backfill function that mirrors run_predict structure"""
    backfill_actuals(cfg, dry_run=dry_run)

if __name__ == "__main__":
    import sys
    from app.utils.env import load_config_env
    
    cfg = load_config_env("config/config.yaml")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        run_backfill(cfg, dry_run=True)
    else:
        run_backfill(cfg, dry_run=False)
