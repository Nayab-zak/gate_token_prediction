#!/usr/bin/env python3
"""
Preprocessing Agent - Clean and prepare data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

class PreprocessingAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "preprocessing_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PreprocessingAgent')
    
    def load_raw_data(self, input_path):
        """Load raw CSV data"""
        try:
            self.logger.info(f"Loading raw data from: {input_path}")
            df = pd.read_csv(input_path)
            self.logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def parse_datetime(self, df):
        """Parse MoveDate and combine with MoveHour to create timestamp"""
        try:
            self.logger.info("Parsing datetime columns")
            
            # Convert MoveDate to datetime with mixed format and dayfirst=True
            df['MoveDate'] = pd.to_datetime(df['MoveDate'], format='mixed', dayfirst=True)
            
            # Ensure MoveHour is integer
            df['MoveHour'] = df['MoveHour'].astype(int)
            
            # Create combined timestamp
            df['timestamp'] = df['MoveDate'] + pd.to_timedelta(df['MoveHour'], unit='h')
            
            self.logger.info("Successfully created timestamp column")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing datetime: {str(e)}")
            raise
    
    def check_correlations(self, df, threshold=0.89):
        """Check correlations between features and target"""
        try:
            self.logger.info(f"Checking correlations with threshold: {threshold}")
            
            # Calculate correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            # Check correlation between ContainerCount and TokenCount
            if 'ContainerCount' in numeric_cols and 'TokenCount' in numeric_cols:
                corr_value = corr_matrix.loc['ContainerCount', 'TokenCount']
                self.logger.info(f"Correlation between ContainerCount and TokenCount: {corr_value:.3f}")
                
                if abs(corr_value) > threshold:
                    self.logger.warning(f"High correlation detected ({corr_value:.3f}) - will drop ContainerCount")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking correlations: {str(e)}")
            return False
    
    def handle_missing_values(self, df):
        """Handle missing values in critical fields"""
        try:
            self.logger.info("Handling missing values")
            
            initial_rows = len(df)
            
            # Check for missing values
            missing_summary = df.isnull().sum()
            self.logger.info(f"Missing values summary:\n{missing_summary}")
            
            # Drop rows with missing critical fields
            critical_fields = ['timestamp', 'TerminalID', 'Desig', 'MoveType', 'TokenCount']
            
            for field in critical_fields:
                if field in df.columns:
                    before_drop = len(df)
                    df = df.dropna(subset=[field])
                    after_drop = len(df)
                    if before_drop != after_drop:
                        self.logger.info(f"Dropped {before_drop - after_drop} rows due to missing {field}")
            
            # For ContainerCount, impute with 0 if it exists and we're keeping it
            if 'ContainerCount' in df.columns:
                missing_containers = df['ContainerCount'].isnull().sum()
                if missing_containers > 0:
                    df['ContainerCount'].fillna(0, inplace=True)
                    self.logger.info(f"Imputed {missing_containers} missing ContainerCount values with 0")
            
            final_rows = len(df)
            self.logger.info(f"Rows after handling missing values: {final_rows} (dropped {initial_rows - final_rows})")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def sanitize_data(self, df, correlation_threshold=0.89):
        """Sanitize data including correlation checks"""
        try:
            self.logger.info("Sanitizing data")
            
            # Check correlations and potentially drop ContainerCount
            should_drop_containers = self.check_correlations(df, correlation_threshold)
            
            if should_drop_containers and 'ContainerCount' in df.columns:
                df = df.drop('ContainerCount', axis=1)
                self.logger.info("Dropped ContainerCount due to high correlation with TokenCount")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error sanitizing data: {str(e)}")
            raise
    
    def enforce_dtypes(self, df):
        """Enforce correct data types"""
        try:
            self.logger.info("Enforcing correct data types")
            
            # Convert categorical columns
            categorical_cols = ['TerminalID', 'Desig', 'MoveType']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    self.logger.info(f"Converted {col} to category")
            
            # Ensure count columns are integers
            count_cols = ['TokenCount']
            if 'ContainerCount' in df.columns:
                count_cols.append('ContainerCount')
                
            for col in count_cols:
                if col in df.columns:
                    df[col] = df[col].astype('int64')
                    self.logger.info(f"Converted {col} to int64")
            
            self.logger.info(f"Final data types:\n{df.dtypes}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error enforcing data types: {str(e)}")
            raise
    
    def save_clean_data(self, df, output_path):
        """Save cleaned data to CSV"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving clean data to: {output_path}")
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} rows to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving clean data: {str(e)}")
            raise
    
    def run(self, input_path, output_path, correlation_threshold=0.89):
        """Main execution method"""
        try:
            self.logger.info("Starting preprocessing")
            
            # Load raw data
            df = self.load_raw_data(input_path)
            
            # Parse datetime
            df = self.parse_datetime(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Sanitize data (including correlation checks)
            df = self.sanitize_data(df, correlation_threshold)
            
            # Enforce correct data types
            df = self.enforce_dtypes(df)
            
            # Save clean data
            self.save_clean_data(df, output_path)
            
            self.logger.info("Preprocessing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocessing Agent')
    parser.add_argument('--input', required=True, help='Input raw CSV file path')
    parser.add_argument('--output', required=True, help='Output clean CSV file path')
    parser.add_argument('--correlation-threshold', type=float, default=0.89, 
                       help='Correlation threshold for feature dropping')
    
    args = parser.parse_args()
    
    agent = PreprocessingAgent()
    success = agent.run(args.input, args.output, args.correlation_threshold)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
