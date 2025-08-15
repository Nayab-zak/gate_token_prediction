# ===============================
# File: agents/feature_engineering_agent.py
# ===============================
"""
Feature Engineering Agent (no event_ts dependency)
--------------------------------------------------
We build leakage-safe features to predict next-hour TokenCount per
(TerminalID, MoveType, Desig), using MoveDate + MoveHour as the time axis.

Assumptions:
- Input rows contain MoveDate (string like MM/DD/YYYY or ISO) and MoveHour (0-23).
- We DO NOT require or persist an 'event_ts'. Internally we derive a local
  timestamp 'ts' from (MoveDate, MoveHour) for ordering and windowing,
  then drop it by default to match user's preference.

Feature groups (all causal):
1) Calendar features from (MoveDate, MoveHour): hour, dow, month, quarter, doy
   + cyclic encodings (sin/cos) and weekend flags.
2) Lags per key: lag_1h, lag_2h, lag_3h, lag_24h, lag_168h.
3) Rolling stats per key: mean/sum/std over 3/6/12/24 hours, zscore_24h.
4) Context: per-terminal totals at each hour, market_share_prev1h, net_flow_prev24.
5) Quality flags: is_gap_prev1h, obs_in_prev24.

Target (training): y = TokenCount(t + H), default H=1 hour.
Output: features saved under data/features/<mode>/dt=YYYY-MM-DD/ as parquet/csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from config import settings
from utils.logging import get_logger
from utils.io import ensure_dir, timestamped_path, get_output_path

logger = get_logger(__name__)

# ----------------------
# Config defaults (env)
# ----------------------
FE_OUTPUT_DIR = Path(getattr(settings, 'FE_OUTPUT_DIR', 'data/features'))
FE_HORIZON_HOURS = int(getattr(settings, 'FE_HORIZON_HOURS', 1))
FE_WINDOWS = [int(x) for x in getattr(settings, 'FE_WINDOWS', '3,6,12,24').split(',')]
FE_KEYS = [c.strip() for c in getattr(settings, 'FE_KEYS', 'TerminalID,MoveType,Desig').split(',')]
FE_OUTPUT_FORMAT = getattr(settings, 'FE_OUTPUT_FORMAT', 'parquet')  # parquet|csv
FE_INPUT_DIR = Path(getattr(settings, 'FE_INPUT_DIR', 'data/clean'))  # source: preprocessing outputs
FE_KEEP_TS = str(getattr(settings, 'FE_KEEP_TS', 'false')).lower() == 'true'

# MoveDate parsing (fallback if preprocessing didn't create MoveDate_dt)
MOVE_DATE_IS_DATE = getattr(settings, 'MOVE_DATE_IS_DATE', False)
MOVE_DATE_FORMAT = getattr(settings, 'MOVE_DATE_FORMAT', 'MM/DD/YYYY')


@dataclass
class FEConfig:
    mode: str
    input_dir: Path
    output_dir: Path
    horizon_h: int
    windows: list[int]
    keys: list[str]
    out_fmt: str


def _discover_inputs(input_dir: Path, mode: str) -> list[Path]:
    base = input_dir / mode
    files = sorted(base.rglob('*.parquet'))
    if not files:
        files = sorted(base.rglob('*.csv'))
    return files


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _concat_inputs(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        df = _read_any(p)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    return out


# -------- Helpers (no event_ts) --------

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Create local hourly timestamp column 'ts' from MoveDate and MoveHour.
    This column is ONLY used internally; it will be dropped unless FE_KEEP_TS=true.
    """
    df = df.copy()
    if 'MoveHour' not in df.columns:
        raise ValueError('MoveHour is required for feature engineering')
    df['MoveHour'] = pd.to_numeric(df['MoveHour'], errors='coerce').fillna(0).astype(int).clip(0, 23)

    if 'MoveDate' not in df.columns:
        raise ValueError('MoveDate is required for feature engineering')
    
    # MoveDate should already be normalized by preprocessing to MM/DD/YYYY format
    base_date = pd.to_datetime(df['MoveDate'], format='%m/%d/%Y', errors='coerce')

    # combine date + hour (naive local time)
    ts = base_date.dt.floor('D') + pd.to_timedelta(df['MoveHour'], unit='h')
    df['ts'] = ts
    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df['ts']
    df['hour'] = ts.dt.hour
    df['dow'] = ts.dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['weekofyear'] = ts.dt.isocalendar().week
    df['month'] = ts.dt.month
    df['quarter'] = ts.dt.quarter
    df['dayofyear'] = ts.dt.dayofyear
    # Cyclic encodings
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_dow'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 366)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 366)
    # Period boundary flags
    df['is_month_start'] = (ts.dt.is_month_start).astype(int)
    df['is_month_end'] = (ts.dt.is_month_end).astype(int)
    return df


def _group_sort(df: pd.DataFrame, keys: list[str]) -> pd.core.groupby.DataFrameGroupBy:
    return df.sort_values('ts').groupby(keys, sort=False, group_keys=False)


def _lag_features(df: pd.DataFrame, keys: list[str], horizon_h: int) -> pd.DataFrame:
    g = _group_sort(df, keys)
    for k in [1, 2, 3, 24, 168]:
        df[f'lag_{k}h'] = g['TokenCount'].shift(k)
    # Target for training
    df['y_target'] = g['TokenCount'].shift(-horizon_h)
    return df


def _rolling_features(df: pd.DataFrame, keys: list[str], windows: list[int]) -> pd.DataFrame:
    g = _group_sort(df, keys)
    for w in windows:
        roll = g['TokenCount'].rolling(window=w, min_periods=max(1, w//2))
        df[f'mean_{w}h'] = roll.mean().reset_index(level=keys, drop=True)
        df[f'std_{w}h'] = roll.std().reset_index(level=keys, drop=True)
        df[f'sum_{w}h'] = roll.sum().reset_index(level=keys, drop=True)
    if 'mean_24h' in df and 'std_24h' in df:
        df['zscore_24h'] = (df['TokenCount'] - df['mean_24h']) / df['std_24h'].replace(0, np.nan)
    return df


def _context_features(df: pd.DataFrame) -> pd.DataFrame:
    # totals per TerminalID at each hour (across MoveType/Desig)
    g_term = df.sort_values('ts').groupby(['TerminalID', 'ts'])
    totals = g_term['TokenCount'].sum().rename('terminal_total_t')
    df = df.join(totals, on=['TerminalID', 'ts'])
    # Market share vs terminal at t-1
    df['market_share_prev1h'] = df.groupby(['TerminalID'])['TokenCount'].shift(1) / df.groupby(['TerminalID'])['terminal_total_t'].shift(1)
    # Net flow last 24h using sign by MoveType
    sign = df['MoveType'].map({'IN': 1, 'OUT': -1}).fillna(0)
    df['signed_tokens'] = sign * df['TokenCount']
    gk = _group_sort(df, ['TerminalID'])
    df['net_flow_prev24'] = gk['signed_tokens'].rolling(window=24, min_periods=12).sum().reset_index(level=['TerminalID'], drop=True)
    df.drop(columns=['signed_tokens'], inplace=True)
    return df


def _quality_flags(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = _group_sort(df, keys)
    prev_ts = g['ts'].shift(1)
    df['is_gap_prev1h'] = ((df['ts'] - prev_ts).dt.components.hours != 1).fillna(1).astype(int)
    df['obs_in_prev24'] = g['ts'].rolling(window=24, min_periods=1).count().reset_index(level=keys, drop=True)
    return df


def build_features(df: pd.DataFrame, *, keys: list[str], windows: list[int], horizon_h: int) -> pd.DataFrame:
    for col in ['TokenCount', 'MoveDate', 'MoveHour']:
        if col not in df.columns and f'{col}_dt' not in df.columns:
            raise AssertionError(f'{col} required (or {col}_dt) for feature engineering')
    
    df = _ensure_ts(df)
    
    # Filter out rows with invalid timestamps
    before_count = len(df)
    df = df[df['ts'].notna()]
    after_count = len(df)
    if before_count != after_count:
        logger.info(f'Filtered out {before_count - after_count} rows with invalid timestamps')
    
    df = _calendar_features(df)
    df = _lag_features(df, keys, horizon_h)
    df = _rolling_features(df, keys, windows)
    df = _context_features(df)
    df = _quality_flags(df, keys)
    return df


def run_feature_engineering() -> Path:
    cfg = FEConfig(
        mode=settings.INGEST_MODE,
        input_dir=FE_INPUT_DIR,
        output_dir=FE_OUTPUT_DIR,
        horizon_h=FE_HORIZON_HOURS,
        windows=FE_WINDOWS,
        keys=FE_KEYS,
        out_fmt=FE_OUTPUT_FORMAT,
    )

    inputs = _discover_inputs(cfg.input_dir, cfg.mode)
    if not inputs:
        raise FileNotFoundError(f'No preprocessed files found under {cfg.input_dir/cfg.mode}')

    df = _concat_inputs(inputs)
    logger.info(f'Loaded {len(df):,} rows for FE')
    feats = build_features(df, keys=cfg.keys, windows=cfg.windows, horizon_h=cfg.horizon_h)

    # Handle missing values differently for training vs realtime
    if cfg.mode == 'realtime':
        # For realtime: Keep all rows, fill missing lag features with 0
        # This allows predictions even with limited historical data
        core_cols = ['TokenCount'] + cfg.keys
        feats = feats.dropna(subset=core_cols)  # Only drop rows missing core features
        
        # Fill missing lag/rolling features with 0 (reasonable for new time series)
        lag_cols = [c for c in feats.columns if c.startswith(('lag_', 'mean_', 'std_', 'sum_', 'zscore_'))]
        feats[lag_cols] = feats[lag_cols].fillna(0)
        
        # Fill other computed features with reasonable defaults
        feats['market_share_prev1h'] = feats['market_share_prev1h'].fillna(0)
        feats['net_flow_prev24'] = feats['net_flow_prev24'].fillna(0)
        feats['is_gap_prev1h'] = feats['is_gap_prev1h'].fillna(1)  # Assume gap if unknown
        feats['obs_in_prev24'] = feats['obs_in_prev24'].fillna(1)
        
        logger.info(f'Realtime mode: Kept all {len(feats):,} rows, filled missing lag features with defaults')
    else:
        # For training: Drop rows with missing lag features (requires clean historical data)
        core_cols = ['TokenCount'] + cfg.keys
        lag_cols = [c for c in feats.columns if c.startswith('lag_')]
        before_count = len(feats)
        feats = feats.dropna(subset=core_cols + lag_cols)
        after_count = len(feats)
        if before_count != after_count:
            logger.info(f'Training mode: Dropped {before_count - after_count} rows with missing lag features')

    logger.info(f'Final feature set: {len(feats):,} rows × {len(feats.columns)} columns')

    # Write output - either partitioned by date or as single consolidated file
    ensure_dir(cfg.output_dir)
    last_path = None
    
    if settings.FE_PARTITION_BY_DATE and not settings.REPLACE_INTERMEDIATE_FILES:
        # Original logic: partition by date (only if not replacing files)
        day_series = pd.to_datetime(feats['ts'], errors='coerce').dt.date.astype(str)
        feats['__day'] = day_series

        for d, part in feats.groupby('__day'):
            out_dir = cfg.output_dir / cfg.mode / f'dt={d}'
            ensure_dir(out_dir)
            out_path = timestamped_path(out_dir, prefix=settings.TABLE_NAME.replace('.', '_'), suffix=f'.{cfg.out_fmt}')
            to_write = part.drop(columns=['__day'] + ([] if FE_KEEP_TS else ['ts']))
            if cfg.out_fmt == 'parquet':
                to_write.to_parquet(out_path, index=False)
            else:
                to_write.to_csv(out_path, index=False)
            last_path = out_path
    else:
        # New logic: create single consolidated file
        table_name_clean = settings.TABLE_NAME.replace('.', '_')
        filename = f'{table_name_clean}_features.{cfg.out_fmt}'
        cleanup_pattern = f'{table_name_clean}_features*.{cfg.out_fmt}' if settings.REPLACE_INTERMEDIATE_FILES else None
        
        out_path = get_output_path(
            base_dir=cfg.output_dir / cfg.mode,
            filename=filename,
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=cleanup_pattern
        )
        
        to_write = feats.drop(columns=([] if FE_KEEP_TS else ['ts']))
        if cfg.out_fmt == 'parquet':
            to_write.to_parquet(out_path, index=False)
        else:
            to_write.to_csv(out_path, index=False)
        last_path = out_path
    assert last_path is not None
    logger.info(f'Wrote features → {last_path}')
    return last_path


if __name__ == '__main__':
    run_feature_engineering()