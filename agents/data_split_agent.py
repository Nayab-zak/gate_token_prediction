#!/usr/bin/env python3
"""
Data Split Agent - Handle chronological data splitting with target variable preparation
"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from datetime import datetime

class DataSplitAgent:
    def __init__(self, config_path="config.yaml", log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "data_split_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataSplitAgent')
        
    def load_wide_data(self, input_path):
        """Load wide format data with proper datetime parsing"""
        try:
            self.logger.info(f"Loading wide format data from: {input_path}")
            df = pd.read_csv(input_path)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} rows with {df.shape[1]} columns")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading wide data: {str(e)}")
            raise
    
    def prepare_target_variables(self, df):
        """Prepare target variables from wide format data"""
        try:
            self.logger.info("Preparing target variables")
            
            # Get all non-timestamp columns (these are our time series)
            series_cols = [col for col in df.columns if col != 'timestamp']
            
            # Create a mapping table for predictions
            target_mapping = []
            for col in series_cols:
                # Parse column name: Terminal_MoveType_Direction
                parts = col.split('_')
                if len(parts) >= 3:
                    terminal_id = parts[0]
                    move_type = parts[1] 
                    direction = '_'.join(parts[2:])  # Handle cases like T/S
                    
                    target_mapping.append({
                        'series_name': col,
                        'TerminalID': terminal_id,
                        'MoveType': move_type, 
                        'Direction': direction
                    })
            
            target_df = pd.DataFrame(target_mapping)
            self.logger.info(f"Created target mapping for {len(target_df)} series")
            
            return target_df, series_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing target variables: {str(e)}")
            raise
    
    def chronological_split_by_date(self, df, train_end_date, val_end_date):
        """Split data chronologically by specific dates"""
        try:
            self.logger.info("Performing chronological split by date")
            
            train_end = pd.to_datetime(train_end_date)
            val_end = pd.to_datetime(val_end_date)
            
            # Create splits
            train_mask = df['timestamp'] < train_end
            val_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < val_end)
            test_mask = df['timestamp'] >= val_end
            
            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()
            test_df = df[test_mask].copy()
            
            self.logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            self.logger.info(f"Date ranges:")
            if len(train_df) > 0:
                self.logger.info(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
            if len(val_df) > 0:
                self.logger.info(f"  Val: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
            if len(test_df) > 0:
                self.logger.info(f"  Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error in chronological split: {str(e)}")
            raise
    
    def save_split_data(self, train_df, val_df, test_df, output_dir):
        """Save split datasets"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            output_paths = {
                'train': output_dir / "wide_train.csv",
                'val': output_dir / "wide_val.csv", 
                'test': output_dir / "wide_test.csv"
            }
            
            for split_name, split_df, out_path in [
                ('train', train_df, output_paths['train']),
                ('val', val_df, output_paths['val']),
                ('test', test_df, output_paths['test'])
            ]:
                split_df.to_csv(out_path, index=False)
                self.logger.info(f"Saved {split_name} split: {out_path} ({len(split_df)} rows)")
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Error saving split data: {str(e)}")
            raise
    
    def run(self, input_path, output_dir):
        """Main execution method"""
        try:
            self.logger.info("Starting data splitting process")
            
            # Load wide format data
            df = self.load_wide_data(input_path)
            
            # Prepare target variable mapping
            target_df, series_cols = self.prepare_target_variables(df)
            
            # Save target mapping
            target_path = Path(output_dir) / "target_mapping.csv"
            target_df.to_csv(target_path, index=False)
            self.logger.info(f"Saved target mapping: {target_path}")
            
            # Chronological split
            train_end_date = self.config['data']['splits']['train_end_date']
            val_end_date = self.config['data']['splits']['val_end_date']
            
            train_df, val_df, test_df = self.chronological_split_by_date(
                df, train_end_date, val_end_date
            )
            
            # Save split data
            output_paths = self.save_split_data(train_df, val_df, test_df, output_dir)
            
            # Save split metadata
            split_info = {
                'train_end_date': train_end_date,
                'val_end_date': val_end_date,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'n_series': len(series_cols),
                'split_timestamp': datetime.now().isoformat()
            }
            
            metadata_path = Path(output_dir) / "split_metadata.yaml"
            with open(metadata_path, 'w') as f:
                yaml.dump(split_info, f, default_flow_style=False)
            
            self.logger.info("Data splitting process completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data splitting process failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Split Agent')
    parser.add_argument('--input-path', required=True, help='Path to wide format CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory for splits')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    agent = DataSplitAgent(args.config)
    success = agent.run(args.input_path, args.output_dir)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
