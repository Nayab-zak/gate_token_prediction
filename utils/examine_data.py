#!/usr/bin/env python3

import pandas as pd
import sys

def examine_excel():
    """Examine the Excel file structure"""
    try:
        # Read the Excel file
        df = pd.read_excel('data/input/moves.xlsx')
        print('Column names:', df.columns.tolist())
        print('\nFirst 5 rows:')
        print(df.head())
        print('\nData types:')
        print(df.dtypes)
        print('\nData shape:', df.shape)
        print('\nSample data for first few columns:')
        for col in df.columns[:5]:
            print(f'\n{col} - unique values: {df[col].nunique()}, sample: {df[col].iloc[:3].tolist()}')
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    examine_excel()
