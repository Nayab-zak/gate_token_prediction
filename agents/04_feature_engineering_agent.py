# Agent for feature engineering
import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from config import DATA_DIR


def setup_logger():
    logger = logging.getLogger('04_feature_engineering_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/04_feature_engineering_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data(split):
    path = os.path.join(DATA_DIR, 'preprocessed', f'{split}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split}.csv not found at {path}")
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.sort_values('datetime')
    return df


def add_calendar_features(df):
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.weekday >= 5
    df['is_friday'] = df['datetime'].dt.weekday == 4
    return df


def add_lag_features(df, max_lag=24):
    for lag in range(1, max_lag+1):
        df[f'lag_{lag}'] = df['TokenCount'].shift(lag)
    return df


def add_rolling_stats(df, windows=[3,6,12,24]):
    for w in windows:
        df[f'roll_mean_{w}'] = df['TokenCount'].rolling(window=w, min_periods=1).mean()
        df[f'roll_std_{w}'] = df['TokenCount'].rolling(window=w, min_periods=1).std().fillna(0)
    df['diff_1'] = df['TokenCount'] - df['TokenCount'].shift(1)
    return df


def add_periodic_encodings(df):
    # sine/cosine for hour and day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


def filter_correlated(df, threshold=0.85, target='TokenCount', logger=None):
    """
    Filter features based on correlation with target.
    
    CRITICAL: This function should ONLY be called on TRAIN data to determine
    which features to keep. The resulting feature list should then be applied
    to test data to ensure consistency.
    
    Args:
        df: DataFrame with features and target
        threshold: Correlation threshold for dropping features
        target: Target variable name
        logger: Logger instance
    
    Returns:
        DataFrame with filtered features
    """
    if logger:
        logger.info(f"Starting correlation filtering with threshold={threshold}")
    
    # Drop explicit non-feature columns
    drop_explicit = ['ContainerCount', 'MoveDate']
    to_drop = [col for col in drop_explicit if col in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        if logger:
            logger.info(f"Dropped explicit non-feature columns: {to_drop}")
    
    # Find features with high correlation to target
    drop_feats = []
    features = [c for c in df.columns if c not in ['TokenCount', 'datetime']]
    
    if logger:
        logger.info(f"Checking correlation for {len(features)} features...")
    
    for feat in features:
        col = df[feat]
        if col.isnull().all():
            drop_feats.append(feat)
            if logger:
                logger.info(f"Dropping {feat}: all null values")
            continue
        
        try:
            r, _ = pearsonr(df[target].fillna(0), col.fillna(0))
            if abs(r) > threshold:
                drop_feats.append(feat)
                if logger:
                    logger.info(f"Dropping {feat}: correlation={r:.4f} (>{threshold})")
        except Exception as e:
            if logger:
                logger.warning(f"Could not compute correlation for {feat}: {e}")
            continue
    
    if drop_feats:
        df = df.drop(columns=drop_feats)
        if logger:
            logger.info(f"✅ Dropped {len(drop_feats)} high-correlation features")
            logger.info(f"Remaining features: {len(df.columns)-2}")  # -2 for datetime, TokenCount
    else:
        if logger:
            logger.info("✅ No features dropped due to correlation")
    
    return df


def process_split(split, logger):
    df = load_data(split)
    logger.info(f"Loaded {split} set with shape {df.shape}")
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_periodic_encodings(df)
    df = filter_correlated(df, logger=logger)

    # drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)

    # Save
    feat_dir = os.path.join(DATA_DIR, 'features')
    os.makedirs(feat_dir, exist_ok=True)
    out_path = os.path.join(feat_dir, f'{split}_features.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {split} features to {out_path}, final shape {df.shape}")


def main():
    logger = setup_logger()
    logger.info("Starting feature engineering...")
    
    # STEP 1: Process train data to determine final feature set
    logger.info("=" * 50)
    logger.info("STEP 1: Processing train data to determine feature set")
    train_df = load_data('train')
    logger.info(f"Loaded train set with shape {train_df.shape}")
    
    # Apply all feature engineering steps to train
    train_df = add_calendar_features(train_df)
    train_df = add_lag_features(train_df)
    train_df = add_rolling_stats(train_df)
    train_df = add_periodic_encodings(train_df)
    
    # Apply correlation filtering ONLY to train data
    logger.info("Applying correlation filtering to determine final feature set...")
    train_df = filter_correlated(train_df, logger=logger)
    
    # Get the final feature columns from train (this is our ground truth)
    final_features = [col for col in train_df.columns if col not in ['datetime', 'TokenCount']]
    logger.info(f"✅ Final feature set determined from train: {len(final_features)} features")
    logger.info(f"Features: {final_features}")
    
    # Drop rows with NaN from lagging
    train_df = train_df.dropna().reset_index(drop=True)
    
    # Save train features
    feat_dir = os.path.join(DATA_DIR, 'features')
    os.makedirs(feat_dir, exist_ok=True)
    train_path = os.path.join(feat_dir, 'train_features.csv')
    train_df.to_csv(train_path, index=False)
    logger.info(f"Saved train features to {train_path}, final shape {train_df.shape}")
    
    # STEP 2: Process test data using the exact same feature set
    logger.info("=" * 50)
    logger.info("STEP 2: Processing test data with consistent feature set")
    test_df = load_data('test')
    logger.info(f"Loaded test set with shape {test_df.shape}")
    
    # Apply all feature engineering steps to test (same as train)
    test_df = add_calendar_features(test_df)
    test_df = add_lag_features(test_df)
    test_df = add_rolling_stats(test_df)
    test_df = add_periodic_encodings(test_df)
    
    # CRITICAL: Do NOT apply filter_correlated to test data
    # Instead, keep only the features that were selected from train
    logger.info("Applying train-determined feature set to test data...")
    keep_cols = ['datetime', 'TokenCount'] + final_features
    available_cols = [col for col in keep_cols if col in test_df.columns]
    missing_cols = [col for col in keep_cols if col not in test_df.columns]
    
    if missing_cols:
        logger.warning(f"⚠️  Features missing in test data: {missing_cols}")
    
    test_df = test_df[available_cols]
    logger.info(f"✅ Test features aligned with train: {len(available_cols)-2} features")
    
    # Drop rows with NaN from lagging
    test_df = test_df.dropna().reset_index(drop=True)
    
    # Save test features
    test_path = os.path.join(feat_dir, 'test_features.csv')
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test features to {test_path}, final shape {test_df.shape}")
    
    # STEP 3: Validation
    logger.info("=" * 50)
    logger.info("STEP 3: Feature consistency validation")
    train_features_final = set(train_df.columns) - {'datetime', 'TokenCount'}
    test_features_final = set(test_df.columns) - {'datetime', 'TokenCount'}
    
    if train_features_final == test_features_final:
        logger.info("✅ SUCCESS: Train and test have identical feature sets")
        logger.info(f"✅ Both datasets have {len(train_features_final)} features")
    else:
        logger.error("❌ MISMATCH: Train and test have different feature sets")
        logger.error(f"Train only: {train_features_final - test_features_final}")
        logger.error(f"Test only: {test_features_final - train_features_final}")
    
    logger.info("Feature engineering completed.")


if __name__ == '__main__':
    main()