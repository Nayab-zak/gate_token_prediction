import pandas as pd
import time
import os
from utils.logger import get_logger

def save_features(df: pd.DataFrame, path: str):
    logger = get_logger('feature_serializer')
    start = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    elapsed = time.time() - start
    size = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Saved features to {path} ({size:.2f} MB) in {elapsed:.2f} sec.")


def load_features(path: str) -> pd.DataFrame:
    logger = get_logger('feature_serializer')
    start = time.time()
    df = pd.read_parquet(path)
    elapsed = time.time() - start
    logger.info(f"Loaded features from {path} ({len(df)} rows) in {elapsed:.2f} sec.")
    return df
