# Agent for data preprocessing
import os
import logging
import pandas as pd

# Attempt to import a holiday library; fallback to config list
try:
    import holidays
    HOLIDAY_LIB_AVAILABLE = True
except ImportError:
    HOLIDAY_LIB_AVAILABLE = False

from config import DATA_DIR, UAE_HOLIDAYS


def setup_logger():
    logger = logging.getLogger('02_preprocess_agent')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs/agents/02_preprocess_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_raw_data():
    raw_dir = os.path.join(DATA_DIR, 'raw')
    dfs = []
    for fname in os.listdir(raw_dir):
        if fname.lower().endswith('.csv'):
            dfs.append(pd.read_csv(os.path.join(raw_dir, fname)))
        elif fname.lower().endswith('.xlsx'):
            dfs.append(pd.read_excel(os.path.join(raw_dir, fname)))
    if not dfs:
        raise FileNotFoundError(f"No raw data files found in {raw_dir}. Please add .csv or .xlsx files.")
    return pd.concat(dfs, ignore_index=True)


def flag_holidays(df):
    if HOLIDAY_LIB_AVAILABLE:
        uae_hols = holidays.CountryHoliday('AE')
        df['is_holiday'] = df['datetime'].dt.date.isin(uae_hols)
    else:
        df['is_holiday'] = df['datetime'].dt.strftime('%Y-%m-%d').isin(UAE_HOLIDAYS)
    return df


def preprocess(df, logger):
    # Ensure MoveDate is string, not datetime
    df['MoveDate'] = df['MoveDate'].astype(str)
    df['MoveHour'] = df['MoveHour'].astype(str)
    df['datetime'] = pd.to_datetime(df['MoveDate'] + ' ' + df['MoveHour'] + ':00')
    # Cast types
    for col in ['MoveType', 'TerminalID', 'Desig']:
        df[col] = df[col].astype('category')
    # Drop missing or invalid
    df = df.dropna(subset=['TokenCount'])
    df = df[df['TokenCount'] >= 0]
    df = df.drop_duplicates(subset=['datetime', 'TerminalID', 'MoveType', 'Desig'])
    # Outlier capping
    mean = df['TokenCount'].mean()
    std = df['TokenCount'].std()
    z = (df['TokenCount'] - mean) / std
    df.loc[z.abs() > 3, 'TokenCount'] = mean + 3 * std
    df['outlier_flag'] = z.abs() > 3
    # Seasonal flags
    df['is_weekend'] = df['datetime'].dt.weekday >= 5
    df['is_friday'] = df['datetime'].dt.weekday == 4
    # Holidays
    df = flag_holidays(df)
    # Save preprocessed
    out_dir = os.path.join(DATA_DIR, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'preprocessed.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved preprocessed data to {out_path}, shape={df.shape}")
    return df


def main():
    logger = setup_logger()
    logger.info("Starting preprocessing...")
    df_raw = load_raw_data()
    df_prep = preprocess(df_raw, logger)
    logger.info("Preprocessing completed.")


if __name__ == '__main__':
    main()
