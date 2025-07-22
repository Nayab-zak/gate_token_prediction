#!/usr/bin/env python3
"""
Feature Agent - Create time-based features, lags, and rolling statistics
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

class FeatureAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "feature_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FeatureAgent')
    
    def load_wide_data(self, input_path):
        """Load wide format data"""
        try:
            self.logger.info(f"Loading wide format data from: {input_path}")
            df = pd.read_csv(input_path)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} rows with {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading wide data: {str(e)}")
            raise
    
    def create_cyclic_features(self, df):
        """Create cyclic time features"""
        try:
            self.logger.info("Creating cyclic time features")
            
            # Hour of day (0-23)
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            
            # Day of week (0-6)
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Day of year (1-366)
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
            df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
            
            # Drop intermediate columns
            df = df.drop(['hour_of_day', 'day_of_week', 'day_of_year'], axis=1)
            
            self.logger.info("Created cyclic time features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating cyclic features: {str(e)}")
            raise
    
    def create_calendar_flags(self, df):
        """Create calendar-based binary flags"""
        try:
            self.logger.info("Creating calendar flags")
            
            # Month end flag
            df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
            
            # Quarter start flag
            df['is_quarter_start'] = df['timestamp'].dt.is_quarter_start.astype(int)
            
            # Summer flag (June, July, August)
            df['is_summer'] = df['timestamp'].dt.month.isin([6, 7, 8]).astype(int)
            
            # Weekend flag
            df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            
            # Business day flag
            df['is_business_day'] = df['timestamp'].dt.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
            
            self.logger.info("Created calendar flags")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating calendar flags: {str(e)}")
            raise
    
    def create_lag_features(self, df, lag_hours=[1, 24, 168]):
        """Create lag features for selected series columns"""
        try:
            self.logger.info(f"Creating lag features for lags: {lag_hours}")
            
            # Get series columns (excluding timestamp and time features)
            time_features = ['timestamp', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                           'doy_sin', 'doy_cos', 'is_month_end', 'is_quarter_start', 
                           'is_summer', 'is_weekend', 'is_business_day']
            
            series_cols = [col for col in df.columns if col not in time_features]
            
            # Select top N series by variance to limit feature explosion
            max_series = 50  # Limit to avoid too many features
            if len(series_cols) > max_series:
                series_variance = df[series_cols].var().sort_values(ascending=False)
                series_cols = series_variance.head(max_series).index.tolist()
                self.logger.info(f"Selected top {max_series} series by variance")
            
            # Create lag features
            for lag in lag_hours:
                self.logger.info(f"Creating {lag}h lag features")
                for col in series_cols:
                    lag_col = f"{col}_lag_{lag}h"
                    df[lag_col] = df[col].shift(lag)
            
            self.logger.info(f"Created lag features for {len(series_cols)} series")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_rolling_features(self, df, windows=[3, 24, 168]):
        """Create rolling statistics"""
        try:
            self.logger.info(f"Creating rolling features for windows: {windows}")
            
            # Get series columns
            time_features = ['timestamp', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                           'doy_sin', 'doy_cos', 'is_month_end', 'is_quarter_start', 
                           'is_summer', 'is_weekend', 'is_business_day']
            
            # Include lag columns in exclusion
            lag_cols = [col for col in df.columns if '_lag_' in col]
            exclude_cols = time_features + lag_cols
            
            series_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Limit series for rolling features
            max_series = 30
            if len(series_cols) > max_series:
                series_variance = df[series_cols].var().sort_values(ascending=False)
                series_cols = series_variance.head(max_series).index.tolist()
                self.logger.info(f"Selected top {max_series} series for rolling features")
            
            # Create rolling features
            for window in windows:
                self.logger.info(f"Creating {window}h rolling features")
                for col in series_cols:
                    # Rolling mean
                    mean_col = f"{col}_roll_mean_{window}h"
                    df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std (only for larger windows to avoid too many features)
                    if window >= 24:
                        std_col = f"{col}_roll_std_{window}h"
                        df[std_col] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            
            self.logger.info(f"Created rolling features for {len(series_cols)} series")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {str(e)}")
            raise
    
    def create_change_features(self, df):
        """Create delta and percentage change features"""
        try:
            self.logger.info("Creating change features")
            
            # Get series columns
            time_features = ['timestamp', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                           'doy_sin', 'doy_cos', 'is_month_end', 'is_quarter_start', 
                           'is_summer', 'is_weekend', 'is_business_day']
            
            # Exclude lag and rolling columns
            feature_cols = [col for col in df.columns if ('_lag_' in col or '_roll_' in col)]
            exclude_cols = time_features + feature_cols
            
            series_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Limit series for change features
            max_series = 20
            if len(series_cols) > max_series:
                series_variance = df[series_cols].var().sort_values(ascending=False)
                series_cols = series_variance.head(max_series).index.tolist()
            
            for col in series_cols:
                # Delta (difference)
                delta_col = f"{col}_delta"
                df[delta_col] = df[col].diff().fillna(0)
                
                # Percentage change
                pct_col = f"{col}_pct_change"
                df[pct_col] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"Created change features for {len(series_cols)} series")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating change features: {str(e)}")
            raise
    
    def save_features(self, df, output_path):
        """Save feature table"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving features to: {output_path}")
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} rows with {df.shape[1]} features to {output_path}")
            
            # Log feature summary
            feature_types = {
                'time_features': len([c for c in df.columns if any(t in c for t in ['sin', 'cos', 'is_'])]),
                'lag_features': len([c for c in df.columns if '_lag_' in c]),
                'rolling_features': len([c for c in df.columns if '_roll_' in c]),
                'change_features': len([c for c in df.columns if ('_delta' in c or '_pct_change' in c)]),
                'original_series': len([c for c in df.columns if c not in ['timestamp'] and not any(t in c for t in ['_lag_', '_roll_', '_delta', '_pct_change', 'sin', 'cos', 'is_'])])
            }
            
            self.logger.info(f"Feature breakdown: {feature_types}")
            
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            raise
    
    def run(self, input_path, output_path, lag_hours=[1, 24, 168], windows=[3, 24, 168]):
        """Main execution method"""
        try:
            self.logger.info("Starting feature engineering")
            
            # Load wide data
            df = self.load_wide_data(input_path)
            
            # Create cyclic time features
            df = self.create_cyclic_features(df)
            
            # Create calendar flags
            df = self.create_calendar_flags(df)
            
            # Create lag features
            df = self.create_lag_features(df, lag_hours)
            
            # Create rolling features
            df = self.create_rolling_features(df, windows)
            
            # Create change features
            df = self.create_change_features(df)
            
            # Save features
            self.save_features(df, output_path)
            
            self.logger.info("Feature engineering completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Agent')
    parser.add_argument('--input', required=True, help='Input wide CSV file path')
    parser.add_argument('--output', required=True, help='Output features CSV file path')
    parser.add_argument('--lag-hours', nargs='+', type=int, default=[1, 24, 168],
                       help='Lag hours for lag features')
    parser.add_argument('--windows', nargs='+', type=int, default=[3, 24, 168],
                       help='Windows for rolling features')
    
    args = parser.parse_args()
    
    agent = FeatureAgent()
    success = agent.run(args.input, args.output, args.lag_hours, args.windows)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
