"""
Compare MLP model results for wide vs combined features.
Scans model metadata files and prints a summary table of test metrics.
"""
import os
import yaml
from glob import glob
from tabulate import tabulate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS_DIR = 'models'
PREDICTIONS_DIR = 'data/predictions'
IMAGES_DIR = 'image'

# Group results by feature set
results = defaultdict(list)
for meta_path in glob(os.path.join(MODELS_DIR, 'mlp*_metadata_*.yaml')):
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    feature_set = meta.get('feature_set', 'wide')
    # Use unique model name for hybrid (combined) features
    if feature_set == 'combined':
        model_name = 'hybrid_mlp'
    else:
        model_name = 'mlp'
    test_metrics = meta.get('test_metrics', {})
    timestamp = meta.get('training_timestamp', '')
    results[model_name].append([
        os.path.basename(meta_path),
        feature_set,
        test_metrics.get('mae', ''),
        test_metrics.get('rmse', ''),
        test_metrics.get('mape', ''),
        timestamp
    ])

for model_name, rows in results.items():
    print(f"\nResults for model: {model_name}")
    print(tabulate(rows, headers=['File', 'Feature Set', 'MAE', 'RMSE', 'MAPE', 'Timestamp']))

if len(results) > 1:
    print("\n✅ Both wide (mlp) and hybrid (hybrid_mlp) models are present. You can compare their results above.")
else:
    print("\n⚠ Only one model type is present. To compare, train/test both feature sets.")

# Collect metrics for bar plot
metric_names = ['mae', 'rmse', 'mape']
bar_data = {m: [] for m in metric_names}
bar_labels = []

for model_name in ['mlp', 'hybrid_mlp']:
    if model_name in results:
        # Take the latest run for each model
        latest = sorted(results[model_name], key=lambda x: x[-1])[-1]
        bar_labels.append(model_name)
        for i, m in enumerate(metric_names, 2):
            try:
                bar_data[m].append(float(latest[i]))
            except:
                bar_data[m].append(np.nan)

# Plot a simple grouped bar chart for metrics (one bar per model per metric)
plt.figure(figsize=(6, 4))
bar_width = 0.35
x = np.arange(len(metric_names))

mlp_vals = [bar_data[m][0] if len(bar_data[m]) > 0 else np.nan for m in metric_names]
hybrid_vals = [bar_data[m][1] if len(bar_data[m]) > 1 else np.nan for m in metric_names]

plt.bar(x - bar_width/2, mlp_vals, bar_width, label='MLP (Wide)', color='tab:blue')
plt.bar(x + bar_width/2, hybrid_vals, bar_width, label='Hybrid MLP (Combined)', color='tab:orange')
plt.xticks(x, [m.upper() for m in metric_names])
plt.ylabel('Score')
plt.title('Test Metrics: MLP vs Hybrid MLP')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'mlp_hybrid_metrics_comparison.png'))
plt.close()

# Collect and plot test predictions vs actuals for both models as a bar chart (average over test set)
actual_means = []
pred_means = []
bar_labels = []

for model_name in ['mlp', 'hybrid_mlp']:
    pred_files = glob(os.path.join(PREDICTIONS_DIR, model_name, f'{model_name}_test_preds_*.csv'))
    if not pred_files:
        continue
    latest_pred = sorted(pred_files)[-1]
    df = pd.read_csv(latest_pred)
    # Compute mean actual and mean predicted
    actual_means.append(df['true_count'].mean())
    pred_means.append(df['pred_count'].mean())
    bar_labels.append(model_name)

plt.figure(figsize=(6, 4))
x = np.arange(len(bar_labels))
bar_width = 0.35
plt.bar(x - bar_width/2, actual_means, bar_width, label='Actual', color='tab:green')
plt.bar(x + bar_width/2, pred_means, bar_width, label='Predicted', color='tab:blue')
plt.xticks(x, [l.upper() for l in bar_labels])
plt.ylabel('Average Count')
plt.title('Test Output: Average Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'mlp_hybrid_test_output_comparison.png'))
plt.close()

print("\nSaved comparison plots:")
print("- image/mlp_hybrid_metrics_comparison.png")
print("- image/mlp_hybrid_test_output_comparison.png")
