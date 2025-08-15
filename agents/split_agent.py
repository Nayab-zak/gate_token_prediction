# ===============================
# File: agents/split_agent.py
# ===============================
"""
Data Splitting & Feature Engineering Agent (NO DATA LEAKAGE)
-------------------------------------------------------------
CORRECT APPROACH: Split FIRST, then Feature Engineering

1. Load preprocessed raw data (no features yet)
2. Split into train/validation/test based on time periods:
   - Test: Latest 6 months of data
   - Validation: 6 months before test period  
   - Train: All data before validation period
3. Run feature engineering SEPARATELY on each split (prevents data leakage)
4. Save 4 parquet files with features:
   - features_train.parquet (training data only)
   - features_valid.parquet (validation data only)  
   - features_test.parquet (test data only)
   - features_train_valid.parquet (combined train + validation)

This ensures rolling windows, statistics, and context features only use 
past data within each split, preventing future information leakage.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import settings
from utils.logging import get_logger
from utils.io import ensure_dir, timestamped_path, get_output_path
from agents.feature_engineering_agent import build_features

logger = get_logger(__name__)

@dataclass
class SplitConfig:
    input_dir: Path
    output_dir: Path
    test_months: int = 6
    valid_months: int = 6
    fe_keys: list[str] = None
    fe_windows: list[int] = None
    fe_horizon_h: int = 1
    output_format: str = 'parquet'


def _read_preprocessed_data(input_dir: Path, mode: str) -> pd.DataFrame:
    """Load all preprocessed CSV files from the input directory.
    
    IMPORTANT: This should load RAW preprocessed data (not features) to prevent data leakage.
    Expected columns: MoveDate, MoveHour, MoveType, TerminalID, Desig, TokenCount
    """
    base = input_dir / mode
    files = sorted(base.rglob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No preprocessed CSV files found under {base}')
    
    parts = []
    for file in files:
        logger.info(f'Loading {file}')
        df = pd.read_csv(file)
        
        # Validate that we have raw data, not features
        expected_cols = {'MoveDate', 'MoveHour', 'MoveType', 'TerminalID', 'Desig', 'TokenCount'}
        actual_cols = set(df.columns)
        
        # Check if this looks like feature data (has lag features, rolling stats, etc.)
        feature_indicators = {'lag_1h', 'lag_24h', 'mean_24h', 'y_target', 'hour', 'dow'}
        if feature_indicators.intersection(actual_cols):
            raise ValueError(
                f"Input data appears to contain features, not raw data! "
                f"Found feature columns: {feature_indicators.intersection(actual_cols)}. "
                f"This would cause data leakage. Please use raw preprocessed data instead."
            )
        
        # Ensure we have the basic required columns
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
        
        parts.append(df)
    
    combined = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    logger.info(f'Loaded total {len(combined):,} rows from {len(files)} files')
    logger.info(f'Columns: {list(combined.columns)}')
    return combined


def _determine_split_dates(df: pd.DataFrame, test_months: int, valid_months: int) -> tuple[datetime, datetime, datetime]:
    """Determine split dates based on the latest data available."""
    # Parse dates and find the latest date in the dataset
    df_temp = df.copy()
    df_temp['parsed_date'] = pd.to_datetime(df_temp['MoveDate'], format='%m/%d/%Y', errors='coerce')
    
    # Remove any rows with invalid dates
    valid_dates = df_temp['parsed_date'].dropna()
    if len(valid_dates) == 0:
        raise ValueError("No valid dates found in the dataset")
    
    latest_date = valid_dates.max()
    earliest_date = valid_dates.min()
    
    # Calculate split boundaries
    test_start = latest_date - timedelta(days=test_months * 30)  # Approximate months as 30 days
    valid_start = test_start - timedelta(days=valid_months * 30)
    
    logger.info(f'Data range: {earliest_date.date()} to {latest_date.date()}')
    logger.info(f'Train period: {earliest_date.date()} to {valid_start.date()}')
    logger.info(f'Valid period: {valid_start.date()} to {test_start.date()}')  
    logger.info(f'Test period: {test_start.date()} to {latest_date.date()}')
    
    return valid_start, test_start, latest_date


def _split_data(df: pd.DataFrame, valid_start: datetime, test_start: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets based on dates."""
    # Parse the MoveDate column
    df_temp = df.copy()
    df_temp['parsed_date'] = pd.to_datetime(df_temp['MoveDate'], format='%m/%d/%Y', errors='coerce')
    
    # Remove rows with invalid dates
    df_clean = df_temp[df_temp['parsed_date'].notna()].copy()
    
    # Create splits
    train_mask = df_clean['parsed_date'] < valid_start
    valid_mask = (df_clean['parsed_date'] >= valid_start) & (df_clean['parsed_date'] < test_start)
    test_mask = df_clean['parsed_date'] >= test_start
    
    train_df = df_clean[train_mask].drop(columns=['parsed_date']).copy()
    valid_df = df_clean[valid_mask].drop(columns=['parsed_date']).copy()
    test_df = df_clean[test_mask].drop(columns=['parsed_date']).copy()
    
    logger.info(f'Split results:')
    logger.info(f'  Train: {len(train_df):,} rows')
    logger.info(f'  Valid: {len(valid_df):,} rows') 
    logger.info(f'  Test: {len(test_df):,} rows')
    
    return train_df, valid_df, test_df


def _save_features(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    """Save the feature dataframe to disk."""
    ensure_dir(output_path.parent)
    
    if output_format.lower() == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    logger.info(f'Saved {len(df):,} rows to {output_path}')


def run_data_splitting() -> dict[str, Path]:
    """Main function to split data and generate features for each split.
    
    This function prevents data leakage by:
    1. Loading raw preprocessed data (no features)
    2. Splitting by time periods FIRST
    3. Building features SEPARATELY for each split
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA SPLITTING + FEATURE ENGINEERING")
    logger.info("This approach prevents data leakage by splitting BEFORE feature engineering")
    logger.info("=" * 60)
    
    cfg = SplitConfig(
        input_dir=Path(settings.PREPROC_OUTPUT_DIR),
        output_dir=Path(settings.FE_OUTPUT_DIR),
        test_months=int(getattr(settings, 'SPLIT_TEST_MONTHS', 6)),
        valid_months=int(getattr(settings, 'SPLIT_VALID_MONTHS', 6)),
        fe_keys=[c.strip() for c in settings.FE_KEYS.split(',')],
        fe_windows=[int(x) for x in settings.FE_WINDOWS.split(',')],
        fe_horizon_h=settings.FE_HORIZON_HOURS,
        output_format=settings.FE_OUTPUT_FORMAT
    )
    
    # Load preprocessed data (raw data, NOT features)
    logger.info("STEP 1: Loading raw preprocessed data...")
    df = _read_preprocessed_data(cfg.input_dir, settings.INGEST_MODE)
    
    # Determine split dates
    logger.info("STEP 2: Determining temporal split boundaries...")
    valid_start, test_start, latest_date = _determine_split_dates(df, cfg.test_months, cfg.valid_months)
    
    # Split the data
    logger.info("STEP 3: Splitting data by time periods...")
    train_df, valid_df, test_df = _split_data(df, valid_start, test_start)
    
    # Build features for each split SEPARATELY (prevents leakage)
    output_paths = {}
    logger.info("STEP 4: Building features separately for each split (NO DATA LEAKAGE)...")
    
    # 1. Train features
    logger.info("Building features for training set...")
    if len(train_df) > 0:
        train_features = build_features(
            train_df, 
            keys=cfg.fe_keys, 
            windows=cfg.fe_windows, 
            horizon_h=cfg.fe_horizon_h
        )
        
        # Drop rows with NA in core/lag features (due to warm-up windows)
        core_cols = ['TokenCount'] + cfg.fe_keys
        lag_cols = [c for c in train_features.columns if c.startswith('lag_')]
        train_features = train_features.dropna(subset=core_cols + lag_cols)
        
        train_path = get_output_path(
            base_dir=cfg.output_dir / settings.INGEST_MODE,
            filename=f'features_train.{cfg.output_format}',
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=f'features_train*.{cfg.output_format}'
        )
        _save_features(train_features, train_path, cfg.output_format)
        output_paths['train'] = train_path
    else:
        logger.warning("No training data available")
    
    # 2. Validation features  
    logger.info("Building features for validation set...")
    if len(valid_df) > 0:
        valid_features = build_features(
            valid_df,
            keys=cfg.fe_keys,
            windows=cfg.fe_windows, 
            horizon_h=cfg.fe_horizon_h
        )
        
        core_cols = ['TokenCount'] + cfg.fe_keys
        lag_cols = [c for c in valid_features.columns if c.startswith('lag_')]
        valid_features = valid_features.dropna(subset=core_cols + lag_cols)
        
        valid_path = get_output_path(
            base_dir=cfg.output_dir / settings.INGEST_MODE,
            filename=f'features_valid.{cfg.output_format}',
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=f'features_valid*.{cfg.output_format}'
        )
        _save_features(valid_features, valid_path, cfg.output_format)
        output_paths['valid'] = valid_path
    else:
        logger.warning("No validation data available")
    
    # 3. Test features
    logger.info("Building features for test set...")
    if len(test_df) > 0:
        test_features = build_features(
            test_df,
            keys=cfg.fe_keys, 
            windows=cfg.fe_windows,
            horizon_h=cfg.fe_horizon_h
        )
        
        core_cols = ['TokenCount'] + cfg.fe_keys
        lag_cols = [c for c in test_features.columns if c.startswith('lag_')]
        test_features = test_features.dropna(subset=core_cols + lag_cols)
        
        test_path = get_output_path(
            base_dir=cfg.output_dir / settings.INGEST_MODE,
            filename=f'features_test.{cfg.output_format}',
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=f'features_test*.{cfg.output_format}'
        )
        _save_features(test_features, test_path, cfg.output_format)
        output_paths['test'] = test_path
    else:
        logger.warning("No test data available")
    
    # 4. Combined train+valid features (for models that don't need separate validation)
    logger.info("Building combined train+valid features...")
    if len(train_df) > 0 and len(valid_df) > 0:
        train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
        train_valid_features = build_features(
            train_valid_df,
            keys=cfg.fe_keys,
            windows=cfg.fe_windows,
            horizon_h=cfg.fe_horizon_h
        )
        
        core_cols = ['TokenCount'] + cfg.fe_keys
        lag_cols = [c for c in train_valid_features.columns if c.startswith('lag_')]
        train_valid_features = train_valid_features.dropna(subset=core_cols + lag_cols)
        
        train_valid_path = get_output_path(
            base_dir=cfg.output_dir / settings.INGEST_MODE,
            filename=f'features_train_valid.{cfg.output_format}',
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=f'features_train_valid*.{cfg.output_format}'
        )
        _save_features(train_valid_features, train_valid_path, cfg.output_format)
        output_paths['train_valid'] = train_valid_path
    elif len(train_df) > 0:
        # If no validation data, just copy train features
        logger.info("No validation data, copying train features to train_valid...")
        output_paths['train_valid'] = output_paths.get('train')
    else:
        logger.warning("No train+valid data available")
    
    logger.info(f"Data splitting and feature engineering completed!")
    logger.info(f"Generated {len(output_paths)} feature files:")
    for split, path in output_paths.items():
        logger.info(f"  {split}: {path}")
    
    return output_paths


if __name__ == '__main__':
    run_data_splitting()
