import pandas as pd
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step

setup_logging()

class SanitizationAgent:
    def __init__(self, df: pd.DataFrame, logger=None):
        """
        Initialize with a DataFrame and optional logger.
        """
        self.df = df.copy()
        self.logger = logger or get_logger('sanitization_agent')

    @track_step('sanitize')
    def sanitize(self, drop_threshold: float = None, fill_strategy: dict = {}):
        """
        Sanitize the DataFrame:
        1. Ensure correct dtypes (datetime for 'MoveDate', numeric for counts).
        2. Drop rows/columns exceeding missing value thresholds.
        3. Fill missing values per fill_strategy.
        4. Cap/remove negative/impossible values.
        5. Log pre/post row counts and return cleaned DataFrame.
        """
        df = self.df
        summary = {'rows_before': len(df)}

        # 1. Ensure correct dtypes
        if 'MoveDate' in df.columns:
            df['MoveDate'] = pd.to_datetime(df['MoveDate'], errors='coerce')
        for col in ['TokenCount', 'ContainerCount', 'MoveHour']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Drop rows/columns exceeding missing value thresholds
        if drop_threshold is not None:
            # Drop columns with too many missing values
            col_thresh = int((1 - drop_threshold) * len(df))
            df = df.dropna(axis=1, thresh=col_thresh)
            # Drop rows with too many missing values
            row_thresh = int((1 - drop_threshold) * len(df.columns))
            before = len(df)
            df = df.dropna(axis=0, thresh=row_thresh)
            self.logger.info(f"Dropped {before - len(df)} rows exceeding missing threshold {drop_threshold}")

        # 3. Fill missing values according to fill_strategy
        for col, strategy in fill_strategy.items():
            if col in df.columns:
                na_count = df[col].isna().sum()
                if pd.api.types.is_numeric_dtype(df[col]):
                    if strategy == 'median':
                        value = df[col].median()
                    elif strategy == 'mean':
                        value = df[col].mean()
                    elif strategy == 'zero':
                        value = 0
                    else:
                        value = strategy
                    df[col] = df[col].fillna(value)
                else:
                    value = strategy if strategy != 'unknown' else 'unknown'
                    df[col] = df[col].fillna(value)
                self.logger.info(f"Filled {na_count} NA in {col} with {value}")

        # 4. Cap or remove negative/impossible values
        for col in ['TokenCount', 'ContainerCount']:
            if col in df.columns:
                before = len(df)
                df = df[df[col] >= 0]
                self.logger.info(f"Removed {before - len(df)} rows with negative values in {col}")

        summary['rows_after'] = len(df)
        self.logger.info(f"Sanitization summary: {summary}")
        return df
