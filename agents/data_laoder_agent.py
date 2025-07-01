# ===== agents/data_loader_agent.py =====
import pandas as pd
from sqlalchemy import create_engine
try:
    from config import INPUT_FILE_PATH, VERTICA_DSN
except ImportError:
    INPUT_FILE_PATH = None
    VERTICA_DSN = None
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step

setup_logging()

class DataLoaderAgent:
    def __init__(self, logger=None):
        self.logger = logger or get_logger('data_loader_agent')

    def load_from_file(self, file_path: str = None) -> pd.DataFrame:
        path = file_path or INPUT_FILE_PATH
        ext = path.lower().split('.')[-1]
        self.logger.info(f"Loading data from file: {path}")
        if ext == 'csv':
            df = pd.read_csv(path, parse_dates=['MoveDate'], dayfirst=True)
        elif ext in ('xls', 'xlsx'):
            df = pd.read_excel(path, parse_dates=['MoveDate'], engine='openpyxl')
        else:
            msg = f"Unsupported file extension: {ext}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info(f"Loaded {len(df)} rows")
        return df

    def load_from_db(self, table: str, dsn: str = None) -> pd.DataFrame:
        dsn = dsn or VERTICA_DSN
        if not dsn:
            msg = 'No DSN provided for DB load'
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info(f"Connecting to DB: {dsn}")
        engine = create_engine(dsn)
        df = pd.read_sql_table(table, engine, parse_dates=['MoveDate'])
        self.logger.info(f"Loaded {len(df)} rows from table {table}")
        return df

    @track_step('data_load')
    def load(self, source: str = 'file', **kwargs) -> pd.DataFrame:
        if source == 'file':
            return self.load_from_file(kwargs.get('file_path'))
        elif source == 'db':
            return self.load_from_db(kwargs.get('table_name'), kwargs.get('dsn'))
        else:
            msg = f"Unknown source: {source}"
            self.logger.error(msg)
            raise ValueError(msg)