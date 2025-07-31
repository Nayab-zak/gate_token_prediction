import os
import pandas as pd
import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Folders to process
folders = [
    'encoded_input',
    'encoded_output',
    'features',
    'final_output',
    'preprocessed'
]

# Round all float columns in all CSVs in the specified folders
def round_csvs():
    for folder in folders:
        dir_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(dir_path):
            continue
        for csv_file in glob.glob(os.path.join(dir_path, '*.csv')):
            try:
                df = pd.read_csv(csv_file)
                float_cols = df.select_dtypes(include=['float']).columns
                if len(float_cols) > 0:
                    df[float_cols] = df[float_cols].round(4)
                    df.to_csv(csv_file, index=False)
                    print(f"Rounded floats in {csv_file}")
                del df  # Explicitly remove DataFrame from memory
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    round_csvs()
