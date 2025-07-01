import pandas as pd
import numpy as np
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step

setup_logging()

class PreprocessingAgent:
    def __init__(self, logger=None):
        self.logger = logger or get_logger('preprocessing_agent')

    @track_step('parse_dates')
    def parse_dates(self, df, cols=['MoveDate']):
        before = len(df)
        for col in cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        after = len(df)
        self.logger.info(f"parse_dates: {before} rows before, {after} after (cols: {cols})")
        return df

    @track_step('normalize_numeric')
    def normalize_numeric(self, df, cols=['TokenCount','ContainerCount'], method='minmax'):
        before = len(df)
        for col in cols:
            if method == 'minmax':
                min_, max_ = df[col].min(), df[col].max()
                df[col] = (df[col] - min_) / (max_ - min_)
            elif method == 'zscore':
                mean, std = df[col].mean(), df[col].std()
                df[col] = (df[col] - mean) / std
        after = len(df)
        self.logger.info(f"normalize_numeric: {before} rows before, {after} after (cols: {cols}, method: {method})")
        return df

    @track_step('handle_outliers')
    def handle_outliers(self, df, cols, method='clip', threshold=3, cap_value=None):
        before = len(df)
        for col in cols:
            z = (df[col] - df[col].mean()) / df[col].std()
            if method == 'clip':
                if cap_value is not None:
                    df[col] = df[col].clip(upper=cap_value)
                else:
                    df[col] = np.where(z.abs() > threshold, np.sign(z) * threshold * df[col].std() + df[col].mean(), df[col])
            elif method == 'remove':
                df = df[z.abs() <= threshold]
        after = len(df)
        self.logger.info(f"handle_outliers: {before} rows before, {after} after (cols: {cols}, method: {method}, threshold: {threshold}, cap_value: {cap_value})")
        return df

    @track_step('encode_missing')
    def encode_missing(self, df, strategy='median'):
        before = len(df)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'median':
                    value = df[col].median()
                elif strategy == 'mean':
                    value = df[col].mean()
                else:
                    value = 0
                df[col] = df[col].fillna(value)
            else:
                df[col] = df[col].fillna('unknown')
        after = len(df)
        self.logger.info(f"encode_missing: {before} rows before, {after} after (strategy: {strategy})")
        return df
