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
    # Handle missing values before processing
    logger.info(f"Original dataframe shape: {df.shape}")
    
    # Check if the dataframe already has a datetime column (from CSV format)
    if 'datetime' in df.columns:
        logger.info("Found existing datetime column, using it directly")
        # Ensure it's in datetime format
        if not pd.api.types.is_datetime64_dtype(df['datetime']):
            logger.info("Converting existing datetime column to datetime type")
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            # Drop any rows where datetime conversion failed
            invalid_dates = df['datetime'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} rows with invalid date formats that couldn't be converted")
                df = df.dropna(subset=['datetime'])
                logger.info(f"Dataframe shape after dropping invalid dates: {df.shape}")
    
    # Handle the case with MoveDate and MoveHour columns (from Excel format)
    elif 'MoveDate' in df.columns and 'MoveHour' in df.columns:
        logger.info("Using MoveDate and MoveHour columns to create datetime")
        
        # Check for missing date/time values
        missing_date = df['MoveDate'].isna().sum()
        missing_hour = df['MoveHour'].isna().sum()
        logger.info(f"Missing MoveDate values: {missing_date}")
        logger.info(f"Missing MoveHour values: {missing_hour}")
        
        # Drop rows with missing date or time values before conversion
        if missing_date > 0 or missing_hour > 0:
            df = df.dropna(subset=['MoveDate', 'MoveHour'])
            logger.info(f"Dropped {missing_date + missing_hour} rows with missing date/time values")
            logger.info(f"Dataframe shape after dropping missing dates: {df.shape}")
        
        # Ensure MoveDate and MoveHour are strings
        df['MoveDate'] = df['MoveDate'].astype(str)
        df['MoveHour'] = df['MoveHour'].astype(str)
        
        # Convert to datetime with error handling
        try:
            df['datetime'] = pd.to_datetime(df['MoveDate'] + ' ' + df['MoveHour'] + ':00', errors='coerce')
            # Drop any rows where datetime conversion failed
            invalid_dates = df['datetime'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} rows with invalid date formats that couldn't be converted")
                df = df.dropna(subset=['datetime'])
                logger.info(f"Dataframe shape after dropping invalid dates: {df.shape}")
        except Exception as e:
            logger.error(f"Error during datetime conversion: {str(e)}")
            raise
    else:
        logger.error("Required date/time columns not found in data. Need either 'datetime' or both 'MoveDate' and 'MoveHour'")
        raise ValueError("Missing required date/time columns in the data")
    
    # Cast types
    for col in ['MoveType', 'TerminalID', 'Desig']:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            logger.warning(f"Column {col} not found in dataframe")
    
    # Drop ContainerCount column as it's not needed
    if 'ContainerCount' in df.columns:
        logger.info("Dropping ContainerCount column as requested")
        df = df.drop(columns=['ContainerCount'])
    
    # Drop missing or invalid token counts
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
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs/agents', exist_ok=True)
        
        # Load and preprocess data
        df_raw = load_raw_data()
        if df_raw.empty:
            logger.error("No raw data found to process")
            return
            
        logger.info(f"Raw data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        
        # Check for required columns
        required_cols = ['TokenCount']
        # If we have a datetime column directly, we don't need MoveDate and MoveHour
        if 'datetime' not in df_raw.columns:
            required_cols.extend(['MoveDate', 'MoveHour'])
        
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return
        
        # Check for ContainerCount column that will be dropped
        if 'ContainerCount' in df_raw.columns:
            logger.info("Found ContainerCount column which will be dropped during preprocessing")
            
        # Show columns for debugging
        logger.info(f"Available columns: {', '.join(df_raw.columns)}")
        
        df_prep = preprocess(df_raw, logger)
        logger.info("Preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
