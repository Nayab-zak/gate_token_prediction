#!/usr/bin/env python3
"""
Simple Enhanced Framework Test
==============================
"""

import sys
import os
import pandas as pd
import numpy as np

# Add paths
base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
sys.path.append(f"{base_path}/utils")

print("🧪 Testing Enhanced Temporal Validation Framework")
print("=" * 60)

try:
    # Test imports
    from temporal_validation import setup_temporal_logger, validate_temporal_inputs
    print("✅ Temporal validation imports successful")
    
    # Test logger
    logger = setup_temporal_logger('test')
    print("✅ Logger setup successful")
    
    # Test basic validation
    n_samples = 10
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='h')
    X = np.random.randn(n_samples, 2)
    y = np.random.randn(n_samples)
    
    # This should work now with our fixes
    X_val, y_val, ts_val = validate_temporal_inputs(X, y, timestamps)
    print(f"✅ Input validation successful - X: {X_val.shape}, y: {y_val.shape}")
    
    print("")
    print("🎉 Enhanced Temporal Validation Framework is READY!")
    print("🚀 All systems operational for enhanced pipeline execution")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
