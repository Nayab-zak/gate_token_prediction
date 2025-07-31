#!/usr/bin/env python3
"""
Enhanced Temporal Validation Framework - Example Usage
====================================================

This script demonstrates how to use the enhanced temporal validation framework
to properly validate time series models without data leakage.

Author: AI Assistant
Date: 2025-07-31
"""

import sys
import os
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from temporal_validation import temporal_cross_validate, generate_validation_report, setup_temporal_logger

def create_sample_time_series_data(n_samples=1000):
    """Create sample time series data for demonstration."""
    
    # Create timestamps (hourly data for ~41 days)
    start_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start_date, periods=n_samples, freq='H')
    
    # Create features with temporal patterns
    np.random.seed(42)
    
    # Feature 1: Hourly pattern
    hour_of_day = timestamps.hour
    feature_1 = np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 2: Daily pattern
    day_of_week = timestamps.dayofweek
    feature_2 = np.sin(2 * np.pi * day_of_week / 7) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 3: Trend
    feature_3 = np.linspace(0, 1, n_samples) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 4: Random noise
    feature_4 = np.random.normal(0, 1, n_samples)
    
    # Create target with realistic relationships
    target = (2 * feature_1 + 
             1.5 * feature_2 + 
             3 * feature_3 + 
             0.5 * feature_4 + 
             np.random.normal(0, 0.2, n_samples))
    
    # Create DataFrame
    X = pd.DataFrame({
        'hourly_pattern': feature_1,
        'daily_pattern': feature_2,
        'trend': feature_3,
        'noise': feature_4
    })
    
    y = pd.Series(target)
    
    return X, y, timestamps

def demonstrate_temporal_validation():
    """Demonstrate the enhanced temporal validation framework."""
    
    print("🚀 Enhanced Temporal Validation Framework Demo")
    print("=" * 60)
    
    # Setup logger
    logger = setup_temporal_logger()
    
    # Create sample data
    logger.info("📊 Creating sample time series data...")
    X, y, timestamps = create_sample_time_series_data(1000)
    
    print(f"📈 Generated {len(X)} samples with {X.shape[1]} features")
    print(f"📅 Time range: {timestamps.min()} to {timestamps.max()}")
    print(f"🎯 Target statistics: mean={y.mean():.3f}, std={y.std():.3f}")
    print()
    
    # Test different models and strategies
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    strategies = ['expanding', 'sliding', 'blocked']
    
    results_summary = {}
    
    for model_name, model in models.items():
        print(f"\n🤖 Testing Model: {model_name}")
        print("-" * 40)
        
        for strategy in strategies:
            print(f"\n📋 Validation Strategy: {strategy}")
            
            try:
                # Perform temporal cross-validation
                cv_results = temporal_cross_validate(
                    model=model,
                    X=X,
                    y=y,
                    timestamps=timestamps,
                    n_splits=5,
                    test_size_ratio=0.2,
                    validation_strategy=strategy,
                    gap_hours=1,  # 1-hour gap to simulate real-world lag
                    logger=logger
                )
                
                # Store results
                key = f"{model_name}_{strategy}"
                results_summary[key] = cv_results
                
                # Print summary
                print(f"✅ {strategy.capitalize()} validation completed:")
                print(f"   RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
                print(f"   MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
                print(f"   MAPE: {cv_results['mape_mean']:.2f}% ± {cv_results['mape_std']:.2f}%")
                print(f"   R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
                print(f"   Stability: {cv_results['performance_stability']}")
                print(f"   Overall: {cv_results['overall_performance']}")
                
            except Exception as e:
                print(f"❌ Error with {strategy} validation: {str(e)}")
                continue
    
    # Generate detailed reports
    print(f"\n📊 GENERATING DETAILED REPORTS")
    print("=" * 60)
    
    report_dir = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/validation_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    for key, results in results_summary.items():
        if results:
            model_name, strategy = key.split('_', 1)
            report_path = f"{report_dir}/temporal_validation_report_{key}.txt"
            
            report = generate_validation_report(results, report_path)
            print(f"📄 Report saved: {report_path}")
    
    # Compare strategies
    print(f"\n🏆 STRATEGY COMPARISON")
    print("=" * 60)
    
    for model_name in models.keys():
        print(f"\n🤖 {model_name} Results:")
        print("-" * 30)
        
        best_rmse = float('inf')
        best_strategy = None
        
        for strategy in strategies:
            key = f"{model_name}_{strategy}"
            if key in results_summary and results_summary[key]:
                rmse = results_summary[key]['rmse_mean']
                stability = results_summary[key]['performance_stability']
                print(f"  {strategy:>10}: RMSE={rmse:.4f}, Stability={stability}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_strategy = strategy
        
        if best_strategy:
            print(f"  🥇 Best: {best_strategy} (RMSE={best_rmse:.4f})")
    
    print(f"\n✅ Temporal validation demonstration completed!")
    print(f"📁 Reports saved in: {report_dir}")
    
    return results_summary

def validate_production_pipeline():
    """Demonstrate how to validate a production pipeline."""
    
    print(f"\n🏭 PRODUCTION PIPELINE VALIDATION")
    print("=" * 60)
    
    # This would be your actual production data loading
    print("📊 Loading production training data...")
    X, y, timestamps = create_sample_time_series_data(2000)
    
    # Production model
    production_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    print("🔍 Performing production-ready temporal validation...")
    
    # Use expanding window (most conservative for production)
    cv_results = temporal_cross_validate(
        model=production_model,
        X=X,
        y=y,
        timestamps=timestamps,
        n_splits=10,  # More folds for robust validation
        test_size_ratio=0.15,  # Smaller test sets
        validation_strategy="expanding",  # Most conservative
        gap_hours=2,  # Realistic prediction lag
        logger=setup_temporal_logger()
    )
    
    # Production readiness check
    print(f"\n🎯 PRODUCTION READINESS ASSESSMENT")
    print("-" * 40)
    
    rmse_mean = cv_results['rmse_mean']
    rmse_std = cv_results['rmse_std']
    stability = cv_results['performance_stability']
    performance = cv_results['overall_performance']
    
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"Stability: {stability}")
    print(f"Performance: {performance}")
    
    # Production deployment decision
    cv_threshold = 0.15  # Example threshold
    stability_ok = stability == 'Stable'
    performance_ok = performance in ['Excellent', 'Good']
    rmse_ok = rmse_mean < cv_threshold
    
    if all([stability_ok, performance_ok, rmse_ok]):
        print(f"✅ MODEL APPROVED FOR PRODUCTION")
        print(f"   ✓ RMSE below threshold ({rmse_mean:.4f} < {cv_threshold})")
        print(f"   ✓ Performance is {performance}")
        print(f"   ✓ Model stability is {stability}")
    else:
        print(f"❌ MODEL NOT READY FOR PRODUCTION")
        if not rmse_ok:
            print(f"   ❌ RMSE too high ({rmse_mean:.4f} >= {cv_threshold})")
        if not performance_ok:
            print(f"   ❌ Performance is {performance}")
        if not stability_ok:
            print(f"   ❌ Model stability is {stability}")
        
        print(f"\n💡 Recommendations:")
        print(f"   • Improve feature engineering")
        print(f"   • Try ensemble methods")
        print(f"   • Increase training data")
        print(f"   • Hyperparameter optimization")
    
    return cv_results

if __name__ == "__main__":
    try:
        # Run temporal validation demonstration
        demo_results = demonstrate_temporal_validation()
        
        # Run production validation example
        prod_results = validate_production_pipeline()
        
        print(f"\n🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
