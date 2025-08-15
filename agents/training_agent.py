from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from config import settings
from utils.io import get_output_path


# ----------------- IO helpers -----------------

def _read_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


# ----------------- target detection -----------------

def _find_target_col(df: pd.DataFrame) -> str:
    # 1) explicit
    if settings.TRAIN_TARGET_COL and settings.TRAIN_TARGET_COL in df.columns:
        return settings.TRAIN_TARGET_COL

    # 2) common names
    candidates = [
        "y_target", "TARGET", "target",
        "TokenCount_tplus1", "TokenCount_Tplus1",
        "TokenCount_plus1h", "TokenCount+1h",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # 3) heuristic
    for c in df.columns:
        if c.lower().endswith("target"):
            return c

    raise ValueError(
        f"Could not find target column in: {list(df.columns)[:40]} ... "
        f"Set TRAIN_TARGET_COL in .env/config."
    )


# ----------------- time utilities (no event_ts) -----------------

def _strftime_map(sql_fmt: str) -> str:
    # SQL-like to Python strftime
    m = {"YYYY": "%Y", "YY": "%y", "MM": "%m", "DD": "%d"}
    py = sql_fmt
    for k, v in m.items():
        py = py.replace(k, v)
    return py

def _parse_move_datetime(df: pd.DataFrame) -> pd.Series:
    """Build local hourly timestamp from MoveDate(+fmt) + MoveHour."""
    if "MoveHour" not in df.columns:
        raise ValueError("MoveHour is required in feature files")

    mh = pd.to_numeric(df["MoveHour"], errors="coerce").fillna(0).astype(int).clip(0, 23)

    if "MoveDate_dt" in df.columns:
        base = pd.to_datetime(df["MoveDate_dt"], errors="coerce")
    elif "MoveDate" in df.columns:
        if getattr(settings, "MOVE_DATE_IS_DATE", False):
            base = pd.to_datetime(df["MoveDate"], errors="coerce")
        else:
            pyfmt = _strftime_map(getattr(settings, "MOVE_DATE_FORMAT", "MM/DD/YYYY"))
            base = pd.to_datetime(df["MoveDate"], format=pyfmt, errors="coerce")
    else:
        raise ValueError("MoveDate or MoveDate_dt must exist in feature files")

    return base.dt.floor("D") + pd.to_timedelta(mh, unit="h")

def _shift_move_datetime(df: pd.DataFrame, horizon_h: int) -> Tuple[pd.Series, pd.Series]:
    """Return (MoveDate_pred, MoveHour_pred) after adding horizon to (MoveDate, MoveHour)."""
    ts = _parse_move_datetime(df)
    ts_pred = ts + pd.to_timedelta(horizon_h, unit="h")
    return ts_pred.dt.date.astype(str), ts_pred.dt.hour.astype(int)


# ----------------- feature prep -----------------

def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[int], str]:
    """Return X, y, cat_feature_indices, target_col. Drops raw date text/ts helpers."""
    tgt = _find_target_col(df)

    # Columns never fed into the model (raw date strings, helper cols, target)
    drop_cols = {
        tgt, "ts", "MoveDate_dt", "__day", "MoveDate"
    }
    drop_existing = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_existing).copy()
    y = df[tgt].astype(float)

    # Poisson requires non-negative targets
    if settings.TRAIN_OBJECTIVE.lower().startswith("poisson"):
        y = y.clip(lower=0.0)

    # Explicit categoricals from config
    cat_cols_cfg = [c.strip() for c in settings.TRAIN_CAT_COLS.split(",") if c.strip()]
    cat_cols_present = [c for c in cat_cols_cfg if c in X.columns]
    for c in cat_cols_present:
        X[c] = X[c].astype("category")

    # Any leftover object columns → treat as categorical too (defensive)
    for c in [c for c in X.columns if X[c].dtype == "object" and c not in cat_cols_present]:
        X[c] = X[c].astype("category")
        cat_cols_present.append(c)

    cat_idx = [X.columns.get_loc(c) for c in cat_cols_present]

    # Replace infs with NaN (CatBoost can handle NaN)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X, y, cat_idx, tgt


# ----------------- metrics & saving -----------------

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.nan) if denom == 0 else float(np.abs(y_true - y_pred).sum() / denom)

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float((2.0 * np.abs(y_pred - y_true) / denom).mean())

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "WAPE": _wape(y_true, y_pred),
        "SMAPE": _smape(y_true, y_pred),
    }

def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _maybe_round_preds(arr: np.ndarray) -> np.ndarray:
    if settings.PRED_ROUND:
        # Business-friendly: non-negative integers
        return np.maximum(0, np.rint(arr)).astype(int)
    return arr

def _save_preds(df_src: pd.DataFrame, y_pred: np.ndarray, path: Path, have_truth: bool, target_col: str | None):
    move_date_pred, move_hour_pred = _shift_move_datetime(df_src, settings.FE_HORIZON_HOURS)
    out = pd.DataFrame({
        "TerminalID": df_src.get("TerminalID"),
        "MoveType":   df_src.get("MoveType"),
        "Desig":      df_src.get("Desig"),
        "MoveDate_pred": move_date_pred,
        "MoveHour_pred": move_hour_pred,
        "TokenCount_pred": _maybe_round_preds(y_pred),
    })
    if have_truth and target_col and target_col in df_src.columns:
        out["TokenCount_true"] = df_src[target_col]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


# ----------------- main -----------------

def main():
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Read feature sets
    df_tr = _read_df(settings.TRAIN_TRAIN_PATH)
    df_va = _read_df(settings.TRAIN_VALID_PATH)
    df_te = _read_df(settings.TRAIN_TEST_PATH)

    # Prepare matrices
    X_tr, y_tr, cat_idx, tgt_tr = _prepare_xy(df_tr)
    X_va, y_va, _,     tgt_va = _prepare_xy(df_va)

    # Test may or may not contain target
    has_test_truth = any(c in df_te.columns for c in {tgt_tr, tgt_va, settings.TRAIN_TARGET_COL, "y_target", "TARGET", "target"})
    drop_cols_test = [c for c in ["ts", "MoveDate_dt", "__day", "MoveDate", tgt_tr, tgt_va, settings.TRAIN_TARGET_COL] if c in df_te.columns]
    X_te = df_te.drop(columns=drop_cols_test).copy()
    # Align dtypes for categoricals
    for idx in cat_idx:
        c = X_tr.columns[idx]
        if c in X_te.columns:
            X_te[c] = X_te[c].astype("category")
    y_te = df_te[_find_target_col(df_te)].astype(float).values if has_test_truth else None
    if has_test_truth and settings.TRAIN_OBJECTIVE.lower().startswith("poisson"):
        y_te = np.clip(y_te, 0.0, None)

    # Model params
    params = {
        "loss_function": settings.TRAIN_OBJECTIVE,
        "eval_metric": "RMSE",
        "learning_rate": settings.TRAIN_LR,
        "iterations": settings.TRAIN_ITERATIONS,
        "depth": settings.TRAIN_DEPTH,
        "l2_leaf_reg": settings.TRAIN_L2,
        "random_seed": settings.TRAIN_SEED,
        "task_type": settings.TRAIN_TASK_TYPE,
        "bootstrap_type": "MVS",
        "early_stopping_rounds": settings.EARLY_STOP,
        "verbose": 100,
    }
    model = CatBoostRegressor(**params)

    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, label=y_va, cat_features=cat_idx)
    test_pool  = Pool(X_te, cat_features=cat_idx)

    # Fit with early stopping on valid
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Predict
    yhat_tr = model.predict(train_pool)
    yhat_va = model.predict(valid_pool)
    yhat_te = model.predict(test_pool)

    # Metrics
    metrics = {
        "train": _metrics(y_tr.values if isinstance(y_tr, pd.Series) else y_tr, yhat_tr),
        "valid": _metrics(y_va.values if isinstance(y_va, pd.Series) else y_va, yhat_va),
    }
    if has_test_truth:
        metrics["test"] = _metrics(y_te, yhat_te)

    # Artifacts - use simple paths to overwrite previous models
    model_path = get_output_path(
        base_dir=model_dir,
        filename="model.cbm",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="model*.cbm"
    )
    model.save_model(str(model_path))
    
    fi = pd.DataFrame({
        "feature": X_tr.columns,
        "importance": model.get_feature_importance(train_pool, type="FeatureImportance"),
    }).sort_values("importance", ascending=False)
    
    fi_path = get_output_path(
        base_dir=model_dir,
        filename="feature_importances.csv",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="feature_importances*.csv"
    )
    fi.to_csv(fi_path, index=False)
    
    metrics_path = get_output_path(
        base_dir=model_dir,
        filename="metrics.json",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="metrics*.json"
    )
    _save_json({
        "params": params,
        "metrics": metrics,
        "train_rows": int(len(df_tr)),
        "valid_rows": int(len(df_va)),
        "test_rows": int(len(df_te)),
        "horizon_hours": settings.FE_HORIZON_HOURS,
        "cat_features": [X_tr.columns[i] for i in cat_idx],
        "target_column_train": tgt_tr,
        "target_column_valid": tgt_va,
    }, metrics_path)

    # Predictions in your desired shape - also use simple paths
    valid_pred_path = get_output_path(
        base_dir=model_dir,
        filename="predictions_valid.csv",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="predictions_valid*.csv"
    )
    _save_preds(df_va, yhat_va, valid_pred_path, have_truth=True, target_col=tgt_va)
    
    test_pred_path = get_output_path(
        base_dir=model_dir,
        filename="predictions_test.csv",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="predictions_test*.csv"
    )
    _save_preds(df_te, yhat_te, test_pred_path, have_truth=has_test_truth,
                target_col=_find_target_col(df_te) if has_test_truth else None)

    print("✅ Training complete.")
    print(f"Model       → {model_path}")
    print(f"Metrics     → {metrics_path}")
    print(f"Importances → {fi_path}")
    print(f"Valid preds → {valid_pred_path}")
    print(f"Test  preds → {test_pred_path}")


if __name__ == "__main__":
    main()
