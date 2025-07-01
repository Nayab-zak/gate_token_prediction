import os
import pandas as pd
from sqlalchemy import create_engine
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step

setup_logging()

class DBLoader:
    def __init__(self, conn_params: dict, table_name: str, logger=None):
        """
        conn_params: dict with keys like user, password, host, port, database, driver
        table_name: name of the table to load
        """
        self.conn_params = conn_params
        self.table_name = table_name
        self.logger = logger or get_logger('db_loader')
        # Pull from env if not in conn_params
        for k in ['user', 'password', 'host', 'database']:
            env_key = f"DB_{k.upper()}"
            if k not in self.conn_params or not self.conn_params[k]:
                self.conn_params[k] = os.getenv(env_key, None)

    @track_step('db_load')
    def load(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Load data from DB with optional date filtering.
        """
        user = self.conn_params.get('user')
        password = self.conn_params.get('password')
        host = self.conn_params.get('host')
        database = self.conn_params.get('database')
        driver = self.conn_params.get('driver', 'postgresql')
        port = self.conn_params.get('port', 5432)
        # Build SQLAlchemy connection string
        dsn = f"{driver}://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(dsn)
        query = f"SELECT * FROM {self.table_name}"
        params = {}
        if start_date and end_date:
            query += " WHERE MoveDate BETWEEN :start_date AND :end_date"
            params = {'start_date': start_date, 'end_date': end_date}
        elif start_date:
            query += " WHERE MoveDate >= :start_date"
            params = {'start_date': start_date}
        elif end_date:
            query += " WHERE MoveDate <= :end_date"
            params = {'end_date': end_date}
        self.logger.info(f"Running query: {query} with params {params}")
        df = pd.read_sql_query(query, engine, params=params, parse_dates=['MoveDate'])
        self.logger.info(f"Loaded {len(df)} rows from {self.table_name}")
        return df
