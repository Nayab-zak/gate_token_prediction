import os
import pandas as pd
import numpy as np
import json

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'final_output')
out_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dashboard_data')
champion_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'champion_model')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(champion_dir, exist_ok=True)

# Step 1: Process all models and calculate metrics
all_metrics = {}

for fname in os.listdir(data_dir):
    if not fname.endswith('_test_results.csv'):
        continue

    # derive the model name
    model_name = fname.replace('_test_results.csv', '')
    print(f"Processing {model_name}...")

    # read and process - maintaining transaction identifiers
    fpath = os.path.join(data_dir, fname)
    df = pd.read_csv(fpath, parse_dates=['datetime'])
    df['prediction'] = np.round(df['prediction']).astype(int)
    
    # compute all errors from scratch
    raw_error = df['prediction'] - df['actual']  # raw error for MSE calculation
    df['error'] = np.abs(raw_error)  # absolute error for CSV output

    # compute metrics correctly using proper formulas
    mse_val = np.round(np.mean(raw_error ** 2), 4)
    rmse_val = np.round(np.sqrt(mse_val), 4)
    mae_val = np.round(np.mean(np.abs(raw_error)), 4)
    mape_val = np.round(np.mean(np.abs(raw_error / df['actual'])) * 100, 4)
    accuracy_val = np.round((df['prediction'] == df['actual']).mean() * 100, 4)
    
    # Store metrics for champion selection
    all_metrics[model_name] = {
        'rmse': float(rmse_val),
        'mae': float(mae_val), 
        'mape': float(mape_val),
        'mse': float(mse_val),
        'accuracy': float(accuracy_val)
    }
    
    print(f"{model_name} - RMSE: {rmse_val}, MAE: {mae_val}, MAPE: {mape_val}%, Accuracy: {accuracy_val}%")

# Step 2: Select champion based on lowest MAPE
print("\n=== Champion Selection ===")
champion_model = None
best_mape = float('inf')

for model_name, metrics in all_metrics.items():
    if metrics['mape'] < best_mape:
        best_mape = metrics['mape']
        champion_model = model_name

champion_stats = all_metrics[champion_model]
print(f"Champion selected: {champion_model} with MAPE: {best_mape}%")

# Step 3: Save champion info
champion_info = {
    'champion': champion_model,
    **champion_stats
}

champion_path = os.path.join(champion_dir, 'champ.json')
with open(champion_path, 'w') as f:
    json.dump(champion_info, f, indent=2)
print(f"Champion info saved to: {champion_path}")

# Step 4: Process CSVs again and add champion info + metrics
print("\n=== Processing CSVs with Champion Info ===")
for fname in os.listdir(data_dir):
    if not fname.endswith('_test_results.csv'):
        continue

    model_name = fname.replace('_test_results.csv', '')
    
    # read and process - maintaining transaction identifiers  
    fpath = os.path.join(data_dir, fname)
    df = pd.read_csv(fpath, parse_dates=['datetime'])
    df['prediction'] = np.round(df['prediction']).astype(int)
    
    # compute errors
    raw_error = df['prediction'] - df['actual']
    df['error'] = np.abs(raw_error)

    # Add metrics as columns (same value for all rows in the dataset)
    model_metrics = all_metrics[model_name]
    df['mse'] = model_metrics['mse']
    df['rmse'] = model_metrics['rmse']
    df['mae'] = model_metrics['mae']
    df['mape'] = model_metrics['mape'] 
    df['accuracy'] = model_metrics['accuracy']
    
    # Add model info
    df['model'] = model_name
    df['active_model'] = champion_model

    # Sort by datetime and save
    df = df.sort_values('datetime').reset_index(drop=True)
    out_path = os.path.join(out_dir, fname)
    df.to_csv(out_path, index=False)
    print(f"Processed and saved: {out_path}")

print(f"\nProcessing completed!")
print(f"Champion model: {champion_model}")
print(f"All CSV files now have model info, metrics, and champion reference.")
