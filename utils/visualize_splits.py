#!/usr/bin/env python3
# filepath: /home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils/visualize_splits.py
"""
Utility script to visualize the time-based train/validation/test split
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR, TEST_SPLIT_MONTHS, VALIDATION_SPLIT_MONTHS

def load_datasets():
    """Load the train, validation and test datasets"""
    base_dir = os.path.join(DATA_DIR, 'preprocessed')
    
    train = pd.read_csv(os.path.join(base_dir, 'train.csv'), parse_dates=['datetime'])
    validation = pd.read_csv(os.path.join(base_dir, 'validation.csv'), parse_dates=['datetime'])
    test = pd.read_csv(os.path.join(base_dir, 'test.csv'), parse_dates=['datetime'])
    
    return train, validation, test

def plot_time_split(train, validation, test):
    """Create a visualization of the time-based split"""
    plt.figure(figsize=(15, 8))
    
    # Create resampled time series for better visualization
    train_daily = train.set_index('datetime')['token_count'].resample('D').mean()
    validation_daily = validation.set_index('datetime')['token_count'].resample('D').mean()
    test_daily = test.set_index('datetime')['token_count'].resample('D').mean()
    
    # Plot data
    plt.plot(train_daily.index, train_daily, 'b-', alpha=0.7, label=f'Train ({len(train)} samples)')
    plt.plot(validation_daily.index, validation_daily, 'g-', alpha=0.7, label=f'Validation ({len(validation)} samples)')
    plt.plot(test_daily.index, test_daily, 'r-', alpha=0.7, label=f'Test ({len(test)} samples)')
    
    # Add vertical lines for split points
    val_cutoff = validation.datetime.min()
    test_cutoff = test.datetime.min()
    
    plt.axvline(x=val_cutoff, color='g', linestyle='--', alpha=0.7, 
                label=f'Validation Cutoff: {val_cutoff.strftime("%Y-%m-%d")}')
    plt.axvline(x=test_cutoff, color='r', linestyle='--', alpha=0.7, 
                label=f'Test Cutoff: {test_cutoff.strftime("%Y-%m-%d")}')
    
    # Add labels and legend
    plt.title(f'Temporal Data Split: Train, Validation ({VALIDATION_SPLIT_MONTHS} months), Test ({TEST_SPLIT_MONTHS} months)', 
              fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Token Count (Daily Average)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    
    # Format x-axis to show dates clearly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Calculate and display split percentages
    total_samples = len(train) + len(validation) + len(test)
    train_pct = len(train) / total_samples * 100
    val_pct = len(validation) / total_samples * 100
    test_pct = len(test) / total_samples * 100
    
    plt.figtext(0.01, 0.01, 
                f'Split Proportions: Train={train_pct:.1f}%, Validation={val_pct:.1f}%, Test={test_pct:.1f}%',
                fontsize=12)
    
    # Save the figure
    output_dir = os.path.join(DATA_DIR, 'dashboard_data')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'temporal_split_visualization.png'), dpi=300)
    
    # Also save in logs for easy reference
    log_dir = os.path.join('logs', 'split_visualizations')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(log_dir, f'temporal_split_{timestamp}.png'), dpi=300)
    
    plt.close()
    
    print(f"Visualization saved to {output_dir}/temporal_split_visualization.png")
    print(f"A copy was also saved to {log_dir}/temporal_split_{timestamp}.png")

def main():
    print("Loading datasets...")
    train, validation, test = load_datasets()
    
    print(f"Train set: {len(train)} rows from {train.datetime.min()} to {train.datetime.max()}")
    print(f"Validation set: {len(validation)} rows from {validation.datetime.min()} to {validation.datetime.max()}")
    print(f"Test set: {len(test)} rows from {test.datetime.min()} to {test.datetime.max()}")
    
    print("\nCreating temporal split visualization...")
    plot_time_split(train, validation, test)
    
    print("\nSummary Statistics:")
    for dataset_name, dataset in [('Train', train), ('Validation', validation), ('Test', test)]:
        print(f"\n{dataset_name} Statistics:")
        print(f"  Date Range: {dataset.datetime.min()} to {dataset.datetime.max()}")
        print(f"  Token Count - Mean: {dataset.token_count.mean():.2f}, Std: {dataset.token_count.std():.2f}")
        print(f"  Token Count - Min: {dataset.token_count.min()}, Max: {dataset.token_count.max()}")

if __name__ == "__main__":
    main()
