#!/usr/bin/env python3
"""
Ingestion Agent - Reads Excel data and converts to raw CSV
"""

import pandas as pd
import logging
import os
from pathlib import Path

class IngestionAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "ingestion_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('IngestionAgent')
    
    def read_excel_to_dataframe(self, input_path):
        """Read Excel file and return DataFrame"""
        try:
            self.logger.info(f"Reading Excel file: {input_path}")
            
            # Check if file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Read all sheets or single sheet
            excel_file = pd.ExcelFile(input_path)
            self.logger.info(f"Found sheets: {excel_file.sheet_names}")
            
            if len(excel_file.sheet_names) == 1:
                df = pd.read_excel(input_path, sheet_name=excel_file.sheet_names[0])
                self.logger.info(f"Read single sheet: {excel_file.sheet_names[0]}")
            else:
                # If multiple sheets, combine them
                dfs = []
                for sheet in excel_file.sheet_names:
                    sheet_df = pd.read_excel(input_path, sheet_name=sheet)
                    dfs.append(sheet_df)
                df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Combined {len(excel_file.sheet_names)} sheets")
            
            self.logger.info(f"DataFrame shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Excel file: {str(e)}")
            raise
    
    def save_raw_csv(self, df, output_path):
        """Save DataFrame to raw CSV"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving raw CSV to: {output_path}")
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully saved {len(df)} rows to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving CSV: {str(e)}")
            raise
    
    def run(self, input_path, output_path):
        """Main execution method"""
        try:
            self.logger.info("Starting ingestion process")
            
            # Read Excel file
            df = self.read_excel_to_dataframe(input_path)
            
            # Save as raw CSV
            self.save_raw_csv(df, output_path)
            
            self.logger.info("Ingestion process completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Ingestion process failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingestion Agent')
    parser.add_argument('--input', required=True, help='Input Excel file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    agent = IngestionAgent()
    success = agent.run(args.input, args.output)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
