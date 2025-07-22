#!/usr/bin/env python3
"""
Scaling Agent - Split data and apply scaling
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class ScalingAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "scaling_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ScalingAgent')
    
    def load_features(self, input_path):
        """Load feature data"""
        try:
            self.logger.info(f"Loading features from: {input_path}")
            df = pd.read_csv(input_path)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} rows with {df.shape[1]} features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading features: {str(e)}")
            raise
    
    def chronological_split(self, df, train_ratio=0.7, val_ratio=0.15):
        """Split data chronologically by timestamp"""
        try:
            self.logger.info("Performing chronological split")
            
            n_samples = len(df)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            self.logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            self.logger.info(f"Date ranges:")
            self.logger.info(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
            self.logger.info(f"  Val: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
            self.logger.info(f"  Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error in chronological split: {str(e)}")
            raise
    
    def prepare_for_scaling(self, df):
        """Prepare data for scaling by separating numeric and non-numeric columns"""
        try:
            # Identify numeric columns (exclude timestamp)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            
            self.logger.info(f"Numeric columns: {len(numeric_cols)}")
            self.logger.info(f"Non-numeric columns: {non_numeric_cols}")
            
            return numeric_cols, non_numeric_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing for scaling: {str(e)}")
            raise
    
    def fit_scaler(self, train_df, numeric_cols, scaler_path):
        """Fit scaler on training data"""
        try:
            self.logger.info("Fitting scaler on training data")
            
            # Initialize scaler
            scaler = StandardScaler()
            
            # Fit on training data
            scaler.fit(train_df[numeric_cols])
            
            # Save scaler
            scaler_dir = Path(scaler_path).parent
            scaler_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(scaler, scaler_path)
            self.logger.info(f"Saved scaler to: {scaler_path}")
            
            return scaler
            
        except Exception as e:
            self.logger.error(f"Error fitting scaler: {str(e)}")
            raise
    
    def transform_and_save(self, df, scaler, numeric_cols, non_numeric_cols, output_path):
        """Transform data and save scaled version"""
        try:
            self.logger.info(f"Transforming and saving to: {output_path}")
            
            # Transform numeric columns
            scaled_numeric = scaler.transform(df[numeric_cols])
            scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_cols, index=df.index)
            
            # Add back non-numeric columns
            for col in non_numeric_cols:
                scaled_df[col] = df[col].values
            
            # Reorder columns to match original
            scaled_df = scaled_df[df.columns]
            
            # Save scaled data
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            scaled_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Saved {len(scaled_df)} rows to {output_path}")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error transforming and saving: {str(e)}")
            raise
    
    def run(self, input_path, output_dir, scaler_path, train_ratio=0.7, val_ratio=0.15):
        """Main execution method"""
        try:
            self.logger.info("Starting scaling process")
            
            # Load features
            df = self.load_features(input_path)
            
            # Chronological split
            train_df, val_df, test_df = self.chronological_split(df, train_ratio, val_ratio)
            
            # Prepare for scaling
            numeric_cols, non_numeric_cols = self.prepare_for_scaling(df)
            
            # Fit scaler
            scaler = self.fit_scaler(train_df, numeric_cols, scaler_path)
            
            # Transform and save splits
            output_paths = {
                'train': Path(output_dir) / "X_train_scaled.csv",
                'val': Path(output_dir) / "X_val_scaled.csv", 
                'test': Path(output_dir) / "X_test_scaled.csv"
            }
            
            for split_name, split_df, out_path in [
                ('train', train_df, output_paths['train']),
                ('val', val_df, output_paths['val']),
                ('test', test_df, output_paths['test'])
            ]:
                self.transform_and_save(split_df, scaler, numeric_cols, non_numeric_cols, out_path)
            
            self.logger.info("Scaling process completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Scaling process failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scaling Agent')
    parser.add_argument('--input', required=True, help='Input features CSV file path')
    parser.add_argument('--output-dir', required=True, help='Output directory for scaled CSVs')
    parser.add_argument('--scaler-path', required=True, help='Path to save scaler object')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation data ratio')
    
    args = parser.parse_args()
    
    agent = ScalingAgent()
    success = agent.run(args.input, args.output_dir, args.scaler_path, 
                       args.train_ratio, args.val_ratio)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
