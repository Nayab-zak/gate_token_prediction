#!/usr/bin/env python3
"""
Simple test of enhanced training for one model.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add paths
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents')

def test_single_model():
    """Test enhanced training for a single model."""
    print("Testing enhanced training for rf_classic...")
    
    try:
        import importlib.util
        
        # Load training agent module
        spec = importlib.util.spec_from_file_location(
            "enhanced_training_agent_updater", 
            "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents/09_enhanced_training_agent_updater.py"
        )
        training_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training_module)
        
        base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
        
        # Test classic model
        print("--- Testing Classic Model ---")
        agent = training_module.EnhancedTrainingAgent('rf', 'classic', base_path)
        X, y, timestamps = agent.load_temporal_data()
        model = agent.create_model()
        print(f"✅ Classic: {X.shape[1]} features, {len(y)} samples, {type(model).__name__}")
        
        # Test augmented model
        print("--- Testing Augmented Model ---")
        agent = training_module.EnhancedTrainingAgent('rf', 'augmented', base_path)
        X, y, timestamps = agent.load_temporal_data()
        model = agent.create_model()
        print(f"✅ Augmented: {X.shape[1]} features, {len(y)} samples, {type(model).__name__}")
        
        print("🎉 Both models test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_model()
    if success:
        print("\n✅ BASIC FUNCTIONALITY WORKS")
    else:
        print("\n❌ BASIC FUNCTIONALITY FAILED")
