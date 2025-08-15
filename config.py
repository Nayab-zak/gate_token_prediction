
# ===============================
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same folder as config.py)
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

class Settings:
    # --- Vertica ---
    VERTICA_HOST: str = os.getenv('VERTICA_HOST', 'localhost')
    VERTICA_PORT: int = int(os.getenv('VERTICA_PORT', '5433'))
    VERTICA_DB: str = os.getenv('VERTICA_DB', '')
    VERTICA_USER: str = os.getenv('VERTICA_USER', '')
    VERTICA_PASSWORD: str = os.getenv('VERTICA_PASSWORD', '')
    VERTICA_TLSMODE: str = os.getenv('VERTICA_TLSMODE', 'disable')  # disable | require | prefer

    # --- Source ---
    TABLE_NAME: str = os.getenv('TABLE_NAME', os.getenv('VERTICA_TABLE_TOKENS', ''))  # tolerate old var name
    DROP_COL_NAME: str | None = os.getenv('DROP_COL_NAME')

    # Data type hints for SQL building
    MOVE_DATE_FORMAT: str = os.getenv('MOVE_DATE_FORMAT', 'MM/DD/YYYY')  # source format of MoveDate if VARCHAR
    MOVE_DATE_IS_DATE: bool = os.getenv('MOVE_DATE_IS_DATE', 'false').lower() == 'true'  # set true if column type is DATE

    # --- Modes: history | realtime ---
    INGEST_MODE: str = os.getenv('INGEST_MODE', 'realtime')

    # --- Common config ---
    TIMEZONE: str = os.getenv('TIMEZONE', 'Asia/Dubai')
    DEDUP_KEY: str = os.getenv('DEDUP_KEY', 'MoveDate,MoveHour,MoveType,TerminalID,Desig')
    BATCH_ROWS: int = int(os.getenv('BATCH_ROWS', '100000'))

    # --- Ingestion & Preprocessing Directories ---
    INGEST_OUTPUT_DIR: str = os.getenv('INGEST_OUTPUT_DIR', 'data/input_raw')
    PREPROC_INPUT_DIR: str = os.getenv('PREPROC_INPUT_DIR', os.path.join(os.getenv('INGEST_OUTPUT_DIR', 'data/input_raw'), 'history'))
    PREPROC_OUTPUT_DIR: str = os.getenv('PREPROC_OUTPUT_DIR', 'data/preprocessed')
    PREPROC_REPORT_DIR: str = os.getenv('PREPROC_REPORT_DIR', 'data/preprocessed/reports')
    PREPROC_OUTPUT_FORMAT: str = os.getenv('PREPROC_OUTPUT_FORMAT', 'csv')
    PREPROC_MAX_FILES: int = int(os.getenv('PREPROC_MAX_FILES', '0'))
    PREPROC_WINSORIZE: bool = os.getenv('PREPROC_WINSORIZE', 'false').lower() == 'true'
    PREPROC_PARTITION_BY_DATE: bool = os.getenv('PREPROC_PARTITION_BY_DATE', 'false').lower() == 'true'

    # --- Feature Engineering ---
    FE_INPUT_DIR: str = os.getenv('FE_INPUT_DIR', os.getenv('PREPROC_OUTPUT_DIR', 'data/preprocessed'))
    FE_OUTPUT_DIR: str = os.getenv('FE_OUTPUT_DIR', 'data/features')
    FE_HORIZON_HOURS: int = int(os.getenv('FE_HORIZON_HOURS', '1'))
    FE_WINDOWS: str = os.getenv('FE_WINDOWS', '3,6,12,24')
    FE_KEYS: str = os.getenv('FE_KEYS', 'TerminalID,MoveType,Desig')
    FE_OUTPUT_FORMAT: str = os.getenv('FE_OUTPUT_FORMAT', 'parquet')
    FE_KEEP_TS: bool = os.getenv('FE_KEEP_TS', 'false').lower() == 'true'
    FE_PARTITION_BY_DATE: bool = os.getenv('FE_PARTITION_BY_DATE', 'false').lower() == 'true'

    # --- Data Splitting ---
    SPLIT_TEST_MONTHS: int = int(os.getenv('SPLIT_TEST_MONTHS', '6'))
    SPLIT_VALID_MONTHS: int = int(os.getenv('SPLIT_VALID_MONTHS', '6'))

    # --- Realtime mode ---
    WINDOW_DAYS: int = int(os.getenv('WINDOW_DAYS', '5'))
    LOOKBACK_HOURS: int = int(os.getenv('LOOKBACK_HOURS', '12'))

    # --- History mode ---
    HISTORY_START: str | None = os.getenv('HISTORY_START')
    HISTORY_END: str | None = os.getenv('HISTORY_END')

        # --- Training paths ---
    TRAIN_TRAIN_PATH = os.getenv("TRAIN_TRAIN_PATH", "data/features/history/features_train.parquet")
    TRAIN_VALID_PATH = os.getenv("TRAIN_VALID_PATH", "data/features/history/features_valid.parquet")
    TRAIN_TEST_PATH  = os.getenv("TRAIN_TEST_PATH",  "data/features/history/features_test.parquet")

    # --- Training params ---
    TRAIN_TARGET_COL = os.getenv("TRAIN_TARGET_COL", "").strip()  # leave empty to auto-detect
    TRAIN_OBJECTIVE  = os.getenv("TRAIN_OBJECTIVE", "Poisson")    # Poisson | RMSE | Tweedie:variance_power=1.5
    TRAIN_TASK_TYPE  = os.getenv("TRAIN_TASK_TYPE", "CPU")        # CPU | GPU
    TRAIN_ITERATIONS = int(os.getenv("TRAIN_ITERATIONS", "2000"))
    TRAIN_LR         = float(os.getenv("TRAIN_LR", "0.05"))
    TRAIN_DEPTH      = int(os.getenv("TRAIN_DEPTH", "8"))
    TRAIN_L2         = float(os.getenv("TRAIN_L2", "3.0"))
    EARLY_STOP       = int(os.getenv("EARLY_STOP", "100"))
    TRAIN_SEED       = int(os.getenv("TRAIN_SEED", "42"))

    # --- CatBoost categorical columns ---
    TRAIN_CAT_COLS   = os.getenv("TRAIN_CAT_COLS", "TerminalID,MoveType,Desig")

    # --- Forecast horizon (used only for output timestamps) ---
    FE_HORIZON_HOURS = int(os.getenv("FE_HORIZON_HOURS", "1"))

    # --- Outputs ---
    MODEL_DIR        = os.getenv("MODEL_DIR", "models/catboost")
    PRED_ROUND       = os.getenv("PRED_ROUND", "true").lower() == "true"  # if true, round & clip >=0 in CSV

    EVAL_PRED_PATH   = os.getenv("EVAL_PRED_PATH", "models/catboost/predictions_test.csv")
    EVAL_REPORT_DIR  = os.getenv("EVAL_REPORT_DIR", "data/_reports/eval")
    EVAL_FILTER_TERMINAL = os.getenv("EVAL_FILTER_TERMINAL", "")  # e.g., T1
    EVAL_FILTER_MOVETYPE = os.getenv("EVAL_FILTER_MOVETYPE", "")  # e.g., IN
    EVAL_FILTER_DESIG    = os.getenv("EVAL_FILTER_DESIG", "")     # e.g., EXP

    # --- Push-back to Vertica ---
    PUSH_TABLE_NAME   = os.getenv("PUSH_TABLE_NAME", "DPW_DL.TBL_GATE_TOKENS_PRED")
    PUSH_CREATE_TABLE = os.getenv("PUSH_CREATE_TABLE", "true").lower() == "true"
    PUSH_UPSERT_MODE  = os.getenv("PUSH_UPSERT_MODE", "merge_values")  # merge_values | insert_only
    PUSH_BATCH_SIZE   = int(os.getenv("PUSH_BATCH_SIZE", "500"))
    PUSH_INCLUDE_TRUE = os.getenv("PUSH_INCLUDE_TRUE", "true").lower() == "true"  # store TokenCount_true if present

    # Predictions file to push (you can override via CLI)
    PREDICTIONS_PATH  = os.getenv("PREDICTIONS_PATH", "models/catboost/predictions_test.csv")

    # --- File Management ---
    # Set to true to replace files instead of creating timestamped versions (saves disk space)
    REPLACE_INTERMEDIATE_FILES: bool = os.getenv("REPLACE_INTERMEDIATE_FILES", "true").lower() == "true"
    KEEP_LAST_N_VERSIONS: int = int(os.getenv("KEEP_LAST_N_VERSIONS", "1"))  # Keep last N versions only




settings = Settings()