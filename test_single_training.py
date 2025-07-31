#!/usr/bin/env python3
"""
Test one model training with temporal validation.
"""

import sys
import os
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents')

def test_single_training():
    """Test temporal training for one model."""
    print("Testing enhanced temporal training for rf_classic...")
    
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
        
        # Test RF classic model with full training
        print("--- Testing RF Classic Full Training ---")
        agent = training_module.EnhancedTrainingAgent('rf', 'classic', base_path)
        model, cv_results, deployment_status = agent.train_enhanced_model()
        
        print(f"✅ Training completed!")
        print(f"   RMSE: {cv_results.get('rmse_mean', 'N/A'):.4f}")
        print(f"   R²: {cv_results.get('r2_mean', 'N/A'):.4f}")
        print(f"   Status: {deployment_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_training()
    if success:
        print("\n🎉 SINGLE MODEL TRAINING WORKS!")
    else:
        print("\n❌ SINGLE MODEL TRAINING FAILED")
