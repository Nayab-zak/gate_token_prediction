#!/usr/bin/env python3
"""
Test Model Training Setup - Verify everything is ready for training
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append('.')

print("🧪 Testing model training setup...")

# Test 1: Check required data files
print("\n1. Checking required data files:")
required_files = [
    'data/preprocessed/moves_wide.csv',
    'data/encoded_input/Z_train.csv',
    'data/encoded_input/Z_val.csv', 
    'data/encoded_input/Z_test.csv',
    'data/preprocessed/X_train_scaled.csv',
    'data/preprocessed/X_val_scaled.csv',
    'data/preprocessed/X_test_scaled.csv'
]

missing_files = []
for file_path in required_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"✓ {file_path} ({size:.1f} MB)")
    else:
        print(f"✗ {file_path} (missing)")
        missing_files.append(file_path)

if missing_files:
    print(f"\n❌ Missing {len(missing_files)} required data files.")
    print("Please run the full encoding pipeline first: ./manage.sh encode-data")
    sys.exit(1)

# Test 2: Check Python packages
print("\n2. Checking Python packages:")
required_packages = [
    ('pandas', 'pd'),
    ('numpy', 'np'), 
    ('scikit-learn', 'sklearn'),
    ('xgboost', 'xgb'),
    ('lightgbm', 'lgb'),
    ('catboost', 'catboost'),
    ('optuna', 'optuna'),
    ('streamlit', 'st'),
    ('plotly', 'plotly'),
    ('yaml', 'yaml')
]

missing_packages = []
for package_name, import_name in required_packages:
    try:
        if import_name == 'sklearn':
            import sklearn
        elif import_name == 'xgb':
            import xgboost as xgb
        elif import_name == 'lgb':
            import lightgbm as lgb
        elif import_name == 'st':
            import streamlit as st
        elif import_name == 'pd':
            import pandas as pd
        elif import_name == 'np':
            import numpy as np
        else:
            __import__(import_name)
        print(f"✓ {package_name}")
    except ImportError as e:
        print(f"✗ {package_name} (not installed: {e})")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n❌ Missing {len(missing_packages)} required packages.")
    print("Install with: pip install " + " ".join(missing_packages))
    sys.exit(1)

# Test 3: Check configuration
print("\n3. Checking configuration:")
if os.path.exists('config.yaml'):
    print("✓ config.yaml exists")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'models', 'tuning', 'streamlit']
        for section in required_sections:
            if section in config:
                print(f"✓ config.{section} section present")
            else:
                print(f"✗ config.{section} section missing")
    except Exception as e:
        print(f"✗ Error reading config.yaml: {e}")
        sys.exit(1)
else:
    print("✗ config.yaml missing")
    sys.exit(1)

# Test 4: Check training agents
print("\n4. Checking training agents:")
training_agents = [
    'agents/base_training_agent.py',
    'agents/train_random_forest.py',
    'agents/train_extra_trees.py',
    'agents/train_xgboost.py', 
    'agents/train_lightgbm.py',
    'agents/train_catboost.py',
    'agents/train_elasticnet.py',
    'agents/train_mlp.py',
    'agents/data_split_agent.py',
    'agents/model_training_orchestrator.py'
]

missing_agents = []
for agent_path in training_agents:
    if os.path.exists(agent_path):
        print(f"✓ {agent_path}")
    else:
        print(f"✗ {agent_path} (missing)")
        missing_agents.append(agent_path)

if missing_agents:
    print(f"\n❌ Missing {len(missing_agents)} training agents.")
    sys.exit(1)

# Test 5: Check directory structure
print("\n5. Checking directory structure:")
required_dirs = [
    'models',
    'logs',
    'data/predictions',
    'agents'
]

for dir_path in required_dirs:
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    print(f"✓ {dir_path}/ directory ready")

# Test 6: Test basic data loading
print("\n6. Testing data loading:")
try:
    # Test encoded data
    z_train = pd.read_csv('data/encoded_input/Z_train.csv')
    print(f"✓ Z_train.csv loaded: {z_train.shape}")
    
    # Test wide data 
    wide_data = pd.read_csv('data/preprocessed/moves_wide.csv')
    print(f"✓ moves_wide.csv loaded: {wide_data.shape}")
    
    # Test scaled data
    x_train = pd.read_csv('data/preprocessed/X_train_scaled.csv')
    print(f"✓ X_train_scaled.csv loaded: {x_train.shape}")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# Test 7: Test imports
print("\n7. Testing agent imports:")
try:
    from agents.base_training_agent import BaseTrainingAgent
    print("✓ BaseTrainingAgent imported")
    
    from agents.data_split_agent import DataSplitAgent  
    print("✓ DataSplitAgent imported")
    
    from agents.model_training_orchestrator import ModelTrainingOrchestrator
    print("✓ ModelTrainingOrchestrator imported")
    
except Exception as e:
    print(f"✗ Error importing agents: {e}")
    sys.exit(1)

# Test 8: Check Streamlit apps
print("\n8. Checking Streamlit apps:")
streamlit_apps = [
    'streamlit_model_results.py',
    'streamlit_model_comparison.py'
]

for app_path in streamlit_apps:
    if os.path.exists(app_path):
        print(f"✓ {app_path}")
    else:
        print(f"✗ {app_path} (missing)")

print("\n🎉 All tests passed! Ready for model training.")
print("\nNext steps:")
print("1. Run individual model: python agents/train_random_forest.py")
print("2. Run all models: python agents/model_training_orchestrator.py") 
print("3. View results: streamlit run streamlit_model_results.py --server.port 8501")
print("4. Compare models: streamlit run streamlit_model_comparison.py --server.port 8502")
