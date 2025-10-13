from __future__ import annotations
import time, json, datetime as dt, os
from typing import Any, Dict
import polars as pl
import joblib, yaml, numpy as np, pandas as pd
from app.db.vertica_client import VerticaClient
from app.data.preparation import canonicalize, assemble_datetime, enforce_contract
from app.data.feature_gen import add_lags, add_rollings, add_calendar
from app.utils.hashing import row_hash, feature_hash
from app.utils.env import load_config_env
from app.logging import get_pipeline_logger, get_model_logger, get_data_logger, get_error_logger

def _to_design_matrix(df: pl.DataFrame, fcfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    data_log = get_data_logger()
    data_log.debug("converting_to_design_matrix_for_prediction", 
                  df_shape=(df.height, df.width))
    
    entity = fcfg["entity_keys"]
    num_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_") or c in ["hour","dow","is_weekend"]]
    cat_cols = entity
    
    data_log.info("prediction_design_matrix_prepared",
                 numeric_features=len(num_cols),
                 categorical_features=len(cat_cols))
    
    X = df.select(num_cols + cat_cols).to_pandas()
    return X, num_cols, cat_cols

def predict_intervals(meta: dict, X_processed):
    """Predict confidence intervals using either quantile models or conformal prediction"""
    model_log = get_model_logger()
    
    m = meta["intervals"]["method"]
    ql, qh = meta["intervals"]["quantiles"]
    
    model_log.debug("computing_prediction_intervals",
                   method=m,
                   quantiles=[ql, qh],
                   samples=len(X_processed))
    
    if m == "quantile" and "quantile_models" in meta["intervals"]:
        ql_path = meta["intervals"]["quantile_models"]["low"]
        qh_path = meta["intervals"]["quantile_models"]["high"]
        
        ql_model = joblib.load(ql_path)
        qh_model = joblib.load(qh_path)
        
        low_preds = ql_model.predict(X_processed).tolist()
        high_preds = qh_model.predict(X_processed).tolist()
        
        model_log.info("quantile_intervals_computed",
                      low_quantile=ql,
                      high_quantile=qh,
                      predictions_count=len(low_preds))
        
        return low_preds, high_preds
        
    elif m == "conformal" and "conformal_abs_err" in meta["intervals"]:
        low = meta["intervals"]["conformal_abs_err"]["low"]
        high = meta["intervals"]["conformal_abs_err"]["high"]
        return [-low]*len(X_processed), [high]*len(X_processed)
    else:
        return [None]*len(X_processed), [None]*len(X_processed)

def run_predict(cfg: Dict[str, Any], dry_run: bool = False):
    pipeline_log = get_pipeline_logger()
    data_log = get_data_logger()
    model_log = get_model_logger()
    error_log = get_error_logger()
    
    t0 = time.time()
    champ_dir = os.path.join(cfg["artifact_dir"], "champion")
    
    pipeline_log.info("prediction_pipeline_started",
                     dry_run=dry_run,
                     champion_dir=champ_dir)
    
    try:
        # Load champion model and metadata
        model_log.info("loading_champion_model")
        model = joblib.load(os.path.join(champ_dir, "model.joblib"))
        pre = joblib.load(os.path.join(champ_dir, "preprocessor.joblib"))
        meta = json.load(open(os.path.join(champ_dir, "run_meta.json")))

        primary_metric = yaml.safe_load(open("config/metrics.yaml"))["policy"]["primary_metric"]
        model_cv_mae = round(meta["metrics_cv"][primary_metric], 2)  # Round to 2 decimal places
        
        model_log.info("champion_model_loaded",
                      model_name=meta.get("model_name", "unknown"),
                      model_version=meta.get("model_version", "unknown"),
                      cv_score=model_cv_mae,
                      primary_metric=primary_metric)

        # Data ingestion
        data_log.info("starting_prediction_data_ingestion")
        dbconf = load_config_env("config/db.yaml")  # Use env var substitution
        client = VerticaClient(dbconf)
        sql = open("db/sql/predict_query.sql","r").read()
        # Use the same time logic as training pipeline - utcnow() but note timezone in config is Asia/Dubai (GMT+4)
        end = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start = end - dt.timedelta(hours=int(cfg["realtime"]["feature_history_hours"]))
        
        # Get prediction frequency from config (should match training)
        prediction_freq = int(cfg["prediction_frequency_hours"])
        
        data_log.info("prediction_time_window_configured", 
                     end_utc=end.isoformat(),
                     start_utc=start.isoformat(),
                     prediction_frequency_hours=prediction_freq,
                     timezone_config=cfg.get("timezone", "not_set"),
                     feature_history_hours=int(cfg["realtime"]["feature_history_hours"]))
        
        rows = client.fetch_df("prod", sql, {"predict_start_ts": start, "predict_end_ts": end})
        data_log.info("production_data_fetched", 
                     rows=rows.height, 
                     time_window=f"{start.isoformat()} to {end.isoformat()}")
        
        if rows.height == 0:
            data_log.warning("no_production_data_available",
                         message="No production data available for the time window",
                         time_window=f"{start.isoformat()} to {end.isoformat()}")
            return
            
    except Exception as e:
        error_log.error("production_data_fetch_failed", 
                       error=str(e), 
                       error_type=type(e).__name__)
        return
        
    # Data preprocessing and feature engineering
    data_log.info("starting_feature_engineering")
    df = canonicalize(rows)
    fcfg = yaml.safe_load(open("config/features.yaml"))
    df = enforce_contract(df, fcfg["required_columns"])
    
    # Use the same horizon-adjusted features as training
    df = add_lags(df, fcfg["entity_keys"], fcfg["target"], fcfg["lag_features"], prediction_freq)
    df = add_rollings(df, fcfg["entity_keys"], fcfg["target"], fcfg["rolling_features"], prediction_freq)
    if fcfg.get("calendar_features", True): df = add_calendar(df)

    feat_cols = [x["name"] for x in fcfg["lag_features"]] + [x["name"] for x in fcfg["rolling_features"]]
    df_feat = df.drop_nulls(subset=feat_cols)
    
    data_log.info("feature_engineering_completed", 
                 after_feature_engineering=df_feat.height,
                 dropped_nulls=df.height - df_feat.height,
                 feature_columns_count=len(feat_cols))

    # Look for most recent data to use as prediction base, considering prediction frequency
    last_hour = int((end - dt.timedelta(hours=prediction_freq)).hour)
    last_date = (end - dt.timedelta(hours=prediction_freq)).date().isoformat()
    
    data_log.info("searching_for_prediction_base", 
                 target_date=last_date, 
                 target_hour=last_hour,
                 current_time=end.isoformat(),
                 prediction_frequency_hours=prediction_freq)
    
    base = df_feat.filter((pl.col("MoveDate")==last_date) & (pl.col("MoveHour")==last_hour))
    
    # If no data for exact target time, use the most recent available hour from today
    if base.height == 0:
        today = end.date().isoformat()
        today_data = df_feat.filter(pl.col("MoveDate")==today)
        
        if today_data.height > 0:
            # Get the most recent hour available for today that's before current time
            current_hour = end.hour
            available_hours = today_data.select("MoveHour").unique().to_series().to_list()
            valid_hours = [h for h in available_hours if h < current_hour]
            
            if valid_hours:
                most_recent_hour = max(valid_hours)
                data_log.info("using_most_recent_available_hour", 
                            original_target_hour=last_hour,
                            using_hour=most_recent_hour,
                            date=today,
                            current_utc_hour=current_hour)
                base = today_data.filter(pl.col("MoveHour")==most_recent_hour)
    
    if base.height == 0:
        data_log.warning("no_entities_for_prediction", 
                        message="No rows found for prediction base; nothing to predict.",
                        target_date=last_date,
                        target_hour=last_hour,
                        prediction_frequency_hours=prediction_freq,
                        available_dates=df_feat.select("MoveDate").unique().to_series().to_list() if df_feat.height > 0 else [],
                        available_hours=df_feat.select("MoveHour").unique().to_series().to_list() if df_feat.height > 0 else [])
        return

    # Generate predictions
    model_log.info("generating_predictions", entities=base.height)
    X_df, num_cols, cat_cols = _to_design_matrix(base, fcfg)
    X_processed = pre.transform(X_df)  # Now returns DataFrame with feature names

    yhat = model.predict(X_processed)
    # Round predictions to whole numbers since we're predicting token counts
    yhat = [round(y) for y in yhat]
    
    low, high = predict_intervals(meta, X_processed)
    # Round confidence intervals to 2 decimal places if they exist
    if low and low[0] is not None:
        low = [round(l, 2) if l is not None else None for l in low]
    if high and high[0] is not None:
        high = [round(h, 2) if h is not None else None for h in high]
    
    runtime_ms = int((time.time() - t0) * 1000)

    out_rows = []
    for i in range(base.height):
        bd = base[i, "MoveDate"]
        bh = int(base[i, "MoveHour"])
        dt0 = dt.datetime.fromisoformat(str(bd)) + dt.timedelta(hours=bh)
        # Predict for prediction_frequency hours ahead instead of just 1 hour
        t_dt = dt0 + dt.timedelta(hours=prediction_freq)
        out_rows.append((t_dt.date().isoformat(), t_dt.hour))

    out_df = base.with_columns([
        pl.Series("MoveDate_t", [d for d,_ in out_rows]),
        pl.Series("MoveHour_t", [h for _,h in out_rows]),
        pl.Series("TokenCount_pred", yhat),
        pl.Series("confidence_or_pred_interval_low", low),
        pl.Series("confidence_or_pred_interval_high", high),
    ])

    # Generate a simple incremental SUR_GKEY for this prediction batch
    # Use timestamp-based approach for uniqueness across different runs
    batch_timestamp = int(dt.datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    batch_sur_gkey = batch_timestamp % 100000  # Keep it small but unique per run
    
    out = pl.DataFrame({
        "MoveDate": out_df["MoveDate_t"],
        "MoveHour": out_df["MoveHour_t"],
        "MoveType": out_df["MoveType"],
        "TerminalID": out_df["TerminalID"],
        "Desig": out_df["Desig"],
        "TokenCount_actual": [None]*out_df.height,  # Will be backfilled later when actuals are available
        "TokenCount_pred": out_df["TokenCount_pred"],
        "primary_metric_value": [model_cv_mae]*out_df.height,  # Changed from model_cv_mae to match schema
        "prediction_ts_utc": [dt.datetime.utcnow()]*out_df.height,
        "model_name": [meta["model_name"]]*out_df.height,
        "model_version": [meta["model_version"]]*out_df.height,
        "inference_runtime_ms": [runtime_ms]*out_df.height,
        "confidence_or_pred_interval_low": out_df["confidence_or_pred_interval_low"],
        "confidence_or_pred_interval_high": out_df["confidence_or_pred_interval_high"],
        # Note: SUR_GKEY is excluded as it's an IDENTITY column in Vertica that gets auto-generated
        "BI_CREATED": [dt.datetime.utcnow()]*out_df.height,
        "BI_UPDATED": [dt.datetime.utcnow()]*out_df.height,  # Will be updated during backfill
        "BI_BATCH_ID": [f"predict_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{runtime_ms}"]*out_df.height,
    })
    keys = ["MoveDate","MoveHour","MoveType","TerminalID","Desig"]
    from app.utils.hashing import row_hash, feature_hash
    data_h = row_hash(out, keys).alias("data_hash")
    feat_h = feature_hash(base, [c for c in base.columns if c not in (["TokenCount"] + keys)])
    out = out.with_columns([
        pl.Series("data_hash", data_h),
        pl.Series("feature_hash", [feat_h]*out.height),
    ])

    rtcfg = yaml.safe_load(open("config/realtime.yaml"))
    upsert_keys = list(cfg["realtime"]["upsert_keys"])
    
    if dry_run:
        pipeline_log.info("dry_run_prediction_output", 
                         rows=len(out), 
                         sample=out.head(2).to_dict())
        return
        
    # Write predictions to database
    pipeline_log.info("writing_predictions_to_database",
                     schema=cfg["realtime"]["dev_schema"],
                     table=cfg["realtime"]["dev_table"],
                     rows=len(out),
                     write_mode=cfg["realtime"]["write_mode"])
    
    client.copy_from_dataframe(out, cfg["realtime"]["dev_schema"], cfg["realtime"]["dev_table"], upsert_keys, cfg["realtime"]["write_mode"], cfg["realtime"]["chunk_size"])

    # Save run metadata
    run_meta = {
        "stage": "predict",
        "rows_out": len(out), 
        "duration_sec": round((time.time()-t0), 3), 
        "model_version": meta["model_version"],
        "prediction_frequency_hours": prediction_freq,
        "entities_predicted": base.height
    }
    os.makedirs(cfg["artifact_dir"], exist_ok=True)
    
    try:
        run_meta_file = os.path.join(cfg["artifact_dir"], f"predict_run_meta_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(run_meta_file, "w") as f:
            json.dump(run_meta, f, indent=2)

        pipeline_log.info("prediction_pipeline_completed", 
                        predictions_generated=len(out), 
                        runtime_ms=runtime_ms, 
                        model_version=meta["model_version"],
                        total_duration_seconds=run_meta["duration_sec"],
                        run_meta_file=run_meta_file)
    except Exception as e:
        error_log.error("prediction_pipeline_failed",
                       error=str(e),
                       error_type=type(e).__name__,
                       total_duration_seconds=round(time.time() - t0, 3))
        raise
