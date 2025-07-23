#!/usr/bin/env python3
"""
Calculate real metrics from prediction CSV files
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics_from_csv(model_name):
    """Calculate real metrics from CSV prediction files"""
    model_dir = Path(f"data/predictions/{model_name}")
    
    # Find test predictions CSV
    test_csv_files = list(model_dir.glob("*_test_preds_*.csv"))
    if not test_csv_files:
        return None
    
    latest_test_csv = max(test_csv_files, key=lambda x: x.stat().st_mtime)
    
    try:
        df = pd.read_csv(latest_test_csv)
        
        true_values = df['true_count'].values
        pred_values = df['pred_count'].values
        
        # Calculate metrics
        mae = mean_absolute_error(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        mape = np.mean(np.abs((true_values - pred_values) / true_values)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        return None

def main():
    models = ['random_forest', 'xgboost', 'lightgbm_dense', 'mlp', 'elasticnet', 'catboost', 'extra_trees']
    
    print("Real Model Metrics:")
    print("=" * 50)
    
    for model in models:
        metrics = calculate_metrics_from_csv(model)
        if metrics:
            print(f"{model}:")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print()

if __name__ == "__main__":
    main()
