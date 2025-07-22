#!/usr/bin/env python3
"""
Aggregation Agent - Group data and create wide format
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from itertools import product

class AggregationAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "aggregation_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AggregationAgent')
    
    def load_clean_data(self, input_path):
        """Load cleaned data"""
        try:
            self.logger.info(f"Loading clean data from: {input_path}")
            df = pd.read_csv(input_path)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert categorical columns
            categorical_cols = ['TerminalID', 'Desig', 'MoveType']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            self.logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading clean data: {str(e)}")
            raise
    
    def group_and_sum(self, df):
        """Group by (timestamp, TerminalID, Desig, MoveType) and sum counts"""
        try:
            self.logger.info("Grouping data and summing counts")
            
            # Group by dimensions and sum TokenCount
            group_cols = ['timestamp', 'TerminalID', 'Desig', 'MoveType']
            
            grouped = df.groupby(group_cols, observed=True).agg({
                'TokenCount': 'sum'
            }).reset_index()
            
            self.logger.info(f"Grouped data shape: {grouped.shape}")
            return grouped
            
        except Exception as e:
            self.logger.error(f"Error grouping data: {str(e)}")
            raise
    
    def create_complete_grid(self, df):
        """Create complete cartesian grid and fill missing with zeros"""
        try:
            self.logger.info("Creating complete cartesian grid")
            
            # Get unique values for each dimension
            timestamps = df['timestamp'].unique()
            terminals = df['TerminalID'].unique()
            desigs = df['Desig'].unique()
            move_types = df['MoveType'].unique()
            
            self.logger.info(f"Grid dimensions - Timestamps: {len(timestamps)}, "
                           f"Terminals: {len(terminals)}, Desigs: {len(desigs)}, "
                           f"MoveTypes: {len(move_types)}")
            
            # Create complete grid
            grid = list(product(timestamps, terminals, desigs, move_types))
            complete_df = pd.DataFrame(grid, columns=['timestamp', 'TerminalID', 'Desig', 'MoveType'])
            
            # Left join with grouped data
            complete_df = complete_df.merge(df, on=['timestamp', 'TerminalID', 'Desig', 'MoveType'], 
                                          how='left')
            
            # Fill missing TokenCount with 0
            complete_df['TokenCount'].fillna(0, inplace=True)
            complete_df['TokenCount'] = complete_df['TokenCount'].astype('int64')
            
            self.logger.info(f"Complete grid shape: {complete_df.shape}")
            return complete_df
            
        except Exception as e:
            self.logger.error(f"Error creating complete grid: {str(e)}")
            raise
    
    def pivot_to_wide(self, df):
        """Pivot to wide format with one column per TerminalID_Desig_MoveType"""
        try:
            self.logger.info("Pivoting to wide format")
            
            # Create combination column
            df['terminal_desig_move'] = (df['TerminalID'].astype(str) + '_' + 
                                       df['Desig'].astype(str) + '_' + 
                                       df['MoveType'].astype(str))
            
            # Pivot
            wide_df = df.pivot_table(
                index='timestamp',
                columns='terminal_desig_move',
                values='TokenCount',
                fill_value=0,
                aggfunc='sum'
            ).reset_index()
            
            # Flatten column names
            wide_df.columns.name = None
            
            self.logger.info(f"Wide format shape: {wide_df.shape}")
            self.logger.info(f"Number of feature columns: {wide_df.shape[1] - 1}")
            
            return wide_df
            
        except Exception as e:
            self.logger.error(f"Error pivoting to wide format: {str(e)}")
            raise
    
    def save_wide_data(self, df, output_path):
        """Save wide format data"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving wide format data to: {output_path}")
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} rows with {df.shape[1]} columns to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving wide data: {str(e)}")
            raise
    
    def run(self, input_path, output_path):
        """Main execution method"""
        try:
            self.logger.info("Starting aggregation process")
            
            # Load clean data
            df = self.load_clean_data(input_path)
            
            # Group and sum
            grouped_df = self.group_and_sum(df)
            
            # Create complete grid
            complete_df = self.create_complete_grid(grouped_df)
            
            # Pivot to wide format
            wide_df = self.pivot_to_wide(complete_df)
            
            # Save wide data
            self.save_wide_data(wide_df, output_path)
            
            self.logger.info("Aggregation process completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Aggregation process failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregation Agent')
    parser.add_argument('--input', required=True, help='Input clean CSV file path')
    parser.add_argument('--output', required=True, help='Output wide CSV file path')
    
    args = parser.parse_args()
    
    agent = AggregationAgent()
    success = agent.run(args.input, args.output)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
