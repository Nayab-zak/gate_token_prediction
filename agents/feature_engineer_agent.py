import pandas as pd
import numpy as np
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step
import holidays
from typing import List, Optional

setup_logging()

class FeatureEngineerAgent:
    def __init__(self, logger=None):
        self.logger = logger or get_logger('feature_engineer_agent')

    @track_step('add_holiday_flags')
    def add_holiday_flags(self, df: pd.DataFrame, country: str = 'UAE') -> pd.DataFrame:
        """Add binary holiday flag for each date in 'MoveDate' column using holidays package or custom UAE logic."""
        if country == 'UAE':
            # Ensure years are valid integers and drop NaNs
            years = [int(y) for y in df['MoveDate'].dt.year.dropna().unique() if not pd.isnull(y)]
            uae_holidays = set([
                pd.Timestamp(year, 1, 1).date() for year in years
            ])
            # Add more fixed or variable dates as needed
            uae_holidays.update([
                pd.Timestamp(year, 12, 2).date() for year in years
            ])
            for year in years:
                uae_holidays.add(pd.Timestamp(year, 4, 10).date())  # Eid al-Fitr (example)
                uae_holidays.add(pd.Timestamp(year, 6, 16).date())  # Eid al-Adha (example)
            df['is_holiday'] = df['MoveDate'].dt.date.isin(uae_holidays)
            self.logger.info(f"Added custom UAE holiday flag for years: {sorted(years)}")
        else:
            try:
                import holidays
                hdays = holidays.country_holidays(country)
                df['is_holiday'] = df['MoveDate'].dt.date.isin(hdays)
                self.logger.info(f"Added holiday flag for country: {country}")
            except (NotImplementedError, AttributeError, KeyError) as e:
                self.logger.warning(f"Country '{country}' not supported for holidays. Setting is_holiday=False. Error: {e}")
                df['is_holiday'] = False
        return df

    @track_step('add_lagged_features')
    def add_lagged_features(self, df: pd.DataFrame, cols: List[str], lags: List[int] = [1,24,168], fill_method: str = 'ffill') -> pd.DataFrame:
        """Add lagged features for specified columns and lags."""
        for col in cols:
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df[col].shift(lag)
                if fill_method:
                    if fill_method == 'ffill':
                        df[lag_col] = df[lag_col].ffill()
                    elif fill_method == 'bfill':
                        df[lag_col] = df[lag_col].bfill()
                    else:
                        df[lag_col] = df[lag_col].fillna(method=fill_method)
        self.logger.info(f"Added lagged features for cols: {cols}, lags: {lags}")
        return df

    @track_step('add_rolling_features')
    def add_rolling_features(self, df: pd.DataFrame, cols: List[str], windows: List[int] = [24,168], aggs: List[str] = ['mean','max']) -> pd.DataFrame:
        """Add rolling window features for specified columns, windows, and aggregations."""
        for col in cols:
            for window in windows:
                for agg in aggs:
                    roll_col = f"{col}_roll{window}_{agg}"
                    df[roll_col] = df[col].rolling(window=window, min_periods=1).agg(agg)
        self.logger.info(f"Added rolling features for cols: {cols}, windows: {windows}, aggs: {aggs}")
        return df

    @track_step('categorical_encoder')
    def categorical_encoder(self, df: pd.DataFrame, cols: List[str], method: str = 'target', min_samples: int = 100) -> pd.DataFrame:
        """Encode categorical columns using target encoding (mean of target for each category)."""
        # Assume 'TokenCount' is the target for encoding
        target = 'TokenCount'
        for col in cols:
            counts = df[col].value_counts()
            valid = counts[counts >= min_samples].index
            means = df[df[col].isin(valid)].groupby(col)[target].mean()
            df[col + '_enc'] = df[col].map(means)
            df[col + '_enc'] = df[col + '_enc'].fillna(df[target].mean())
        self.logger.info(f"Encoded categoricals: {cols} using {method} encoding")
        return df

    @track_step('feature_pipeline')
    def feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline and return final feature matrix."""
        # Example pipeline: add holiday, lagged, rolling, and encode categoricals
        df = self.add_holiday_flags(df)
        df = self.add_lagged_features(df, cols=['TokenCount','ContainerCount'])
        df = self.add_rolling_features(df, cols=['TokenCount','ContainerCount'])
        df = self.categorical_encoder(df, cols=['VesselType','Port'])
        self.logger.info("Feature pipeline complete.")
        return df
