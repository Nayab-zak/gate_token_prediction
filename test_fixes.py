#!/usr/bin/env python3
"""
Test fixes for enhanced temporal validation pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents')

def test_data_loading():
    """Test data loading for both classic and augmented variants."""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    # Test classic data
    classic_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/data/features/train_features.csv"
    if os.path.exists(classic_path):
        try:
            data = pd.read_csv(classic_path, nrows=1000)  # Sample only
            feature_cols = [col for col in data.columns 
                           if col not in ['datetime', 'TokenCount', 'TerminalID', 'MoveType', 'Desig']]
            print(f"✅ Classic features: {len(feature_cols)} columns")
            print(f"   Sample shape: {data.shape}")
            print(f"   Target range: {data['TokenCount'].min():.2f} - {data['TokenCount'].max():.2f}")
        except Exception as e:
            print(f"❌ Classic data error: {e}")
    else:
        print("❌ Classic data file not found")
    
    # Test augmented data
    augmented_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/data/encoded_input/train_input_augmented.csv"
    if os.path.exists(augmented_path):
        try:
            data = pd.read_csv(augmented_path, nrows=1000)  # Sample only
            feature_cols = [col for col in data.columns 
                           if col not in ['datetime', 'TokenCount', 'TerminalID', 'MoveType', 'Desig']]
            print(f"✅ Augmented features: {len(feature_cols)} columns")
            print(f"   Sample shape: {data.shape}")
            print(f"   Target range: {data['TokenCount'].min():.2f} - {data['TokenCount'].max():.2f}")
            print(f"   Feature types: {[col for col in feature_cols[:5]]}...")
        except Exception as e:
            print(f"❌ Augmented data error: {e}")
    else:
        print("❌ Augmented data file not found")

def test_temporal_validation():
    """Test temporal validation framework with simple data."""
    print("\n" + "=" * 60)
    print("TESTING TEMPORAL VALIDATION")
    print("=" * 60)
    
    try:
        from temporal_validation import enhanced_temporal_cross_validation, setup_temporal_logger
        
        # Create simple test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 10 + 50
        timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='H')
        
        # Create simple model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        logger = setup_temporal_logger()
        print("✅ Temporal validation components loaded")
        print(f"   Test data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test validation
        results = enhanced_temporal_cross_validation(
            X=X, y=y, timestamps=timestamps, 
            model=model, n_splits=3, 
            strategy='expanding', test_ratio=0.2,
            logger=logger
        )
        
        print("✅ Temporal validation test completed")
        print(f"   RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"   R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        
    except Exception as e:
        print(f"❌ Temporal validation error: {e}")
        import traceback
        traceback.print_exc()

def test_model_selection():
    """Test robust model selection function."""
    print("\n" + "=" * 60)
    print("TESTING MODEL SELECTION")
    print("=" * 60)
    
    try:
        from robust_model_selection import select_best_model_with_confidence
        
        # Create mock model performance data
        model_performances = {
            'rf_classic': {
                'test_rmse': 5.2,
                'validation_rmse': 5.0,
                'test_r2': 0.85,
                'stability': 'HIGH'
            },
            'xgb_classic': {
                'test_rmse': 4.8,
                'validation_rmse': 4.9,
                'test_r2': 0.87,
                'stability': 'HIGH'
            },
            'lgbm_classic': {
                'test_rmse': 5.5,
                'validation_rmse': 5.3,
                'test_r2': 0.82,
                'stability': 'MEDIUM'
            }
        }
        
        results = select_best_model_with_confidence(model_performances)
        
        print("✅ Model selection test completed")
        print(f"   Champion: {results['champion_model']}")
        print(f"   Champion RMSE: {results['champion_metric_value']:.4f}")
        print(f"   Improvement: {results['improvement_over_second']:.2%}")
        print(f"   Significant: {results['is_statistically_significant']}")
        
    except Exception as e:
        print(f"❌ Model selection error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🧪 TESTING ENHANCED PIPELINE FIXES")
    print("=" * 80)
    print(f"Test started: {datetime.now()}")
    
    test_data_loading()
    test_temporal_validation()
    test_model_selection()
    
    print("\n" + "=" * 80)
    print("🎉 TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
