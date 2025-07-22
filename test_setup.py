#!/usr/bin/env python3
"""Simple test script"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append('.')

print("🧪 Testing pipeline components...")

# Test 1: Import agents
try:
    from agents.ingestion_agent import IngestionAgent
    print("✓ Ingestion agent imported successfully")
except Exception as e:
    print(f"✗ Error importing ingestion agent: {e}")

# Test 2: Check input file
try:
    if os.path.exists('data/input/moves.xlsx'):
        df = pd.read_excel('data/input/moves.xlsx')
        print(f"✓ Input file readable: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Columns: {list(df.columns)}")
    else:
        print("✗ Input file not found")
except Exception as e:
    print(f"✗ Error reading input file: {e}")

# Test 3: Create directories
try:
    os.makedirs('data/preprocessed', exist_ok=True)
    os.makedirs('models', exist_ok=True) 
    os.makedirs('logs', exist_ok=True)
    print("✓ Required directories created")
except Exception as e:
    print(f"✗ Error creating directories: {e}")

print("🎉 Basic tests completed!")
