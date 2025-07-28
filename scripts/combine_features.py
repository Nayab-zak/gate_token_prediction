"""
Combine wide and autoencoded features for train/val/test splits.
Saves combined features as combined_train.csv, combined_val.csv, combined_test.csv in data/preprocessed/.
"""
import pandas as pd
import os

# Paths
WIDE_PATH = 'data/preprocessed/wide_{}.csv'
ENCODED_PATH = 'data/encoded_input/Z_{}.csv'
OUT_PATH = 'data/preprocessed/combined_{}.csv'

splits = ['train', 'val', 'test']

for split in splits:
    wide = pd.read_csv(WIDE_PATH.format(split))
    encoded = pd.read_csv(ENCODED_PATH.format(split))
    # Merge on timestamp
    combined = pd.merge(wide, encoded, on='timestamp', how='inner')
    combined.to_csv(OUT_PATH.format(split), index=False)
    print(f"Combined features for {split} saved to {OUT_PATH.format(split)}")
