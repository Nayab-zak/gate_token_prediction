from __future__ import annotations
import yaml
from typing import Dict
from app.utils.schema import load_sink_contract
from app.db.vertica_client import VerticaClient
from app.utils.env import load_config_env
from app.logging import get_db_logger, get_error_logger

TYPE_MAP = {
    "DATE": "DATE",
    "INT": "INT",
    "FLOAT": "FLOAT",
    "TIMESTAMP": "TIMESTAMP",
    "VARCHAR(64)": "VARCHAR(64)"
}

def _emit_create(schema: str, table: str, cols: Dict[str, str]) -> str:
    # Handle special column types
    col_definitions = []
    for c, t in cols.items():
        # All columns are now regular columns - no IDENTITY auto-increment
        col_definitions.append(f"{c} {TYPE_MAP.get(t, t)}")
    
    colsql = ",\n  ".join(col_definitions)
    pk = "(MoveDate, MoveHour, MoveType, TerminalID, Desig)"
    
    return f"""CREATE TABLE IF NOT EXISTS {schema}.{table} (
  {colsql}
);
CREATE PROJECTION IF NOT EXISTS {schema}.{table}_uniq AS SELECT * FROM {schema}.{table} ORDER BY {pk} SEGMENTED BY HASH {pk} ALL NODES;
"""

def _get_missing_columns(vc, schema: str, table: str, expected_cols: Dict[str, str]) -> Dict[str, str]:
    """Check which columns are missing from the existing table"""
    try:
        # Query to get existing columns
        check_sql = f"""
        SELECT column_name 
        FROM columns 
        WHERE table_schema = '{schema}' 
        AND table_name = '{table}'
        """
        result = vc.fetch_df("dev", check_sql, {})
        existing_cols = set(result['column_name'].to_list()) if result.height > 0 else set()
        
        # Find missing columns
        missing = {col: dtype for col, dtype in expected_cols.items() if col not in existing_cols}
        return missing
    except Exception:
        # If we can't check (table doesn't exist), return all columns
        return expected_cols

def _add_missing_columns(vc, schema: str, table: str, missing_cols: Dict[str, str]):
    """Add missing columns to existing table"""
    db_log = get_db_logger()
    error_log = get_error_logger()
    
    for col, dtype in missing_cols.items():
        try:
            alter_sql = f"ALTER TABLE {schema}.{table} ADD COLUMN {col} {TYPE_MAP.get(dtype, dtype)}"
            vc.execute_dev_sql(alter_sql)
            db_log.info("column_added", 
                       table=f"{schema}.{table}", 
                       column=col, 
                       column_type=dtype)
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                db_log.info("column_already_exists", 
                           table=f"{schema}.{table}", 
                           column=col)
            else:
                error_log.error("column_add_failed", 
                               table=f"{schema}.{table}", 
                               column=col, 
                               error=str(e),
                               error_type=type(e).__name__)
                raise

def create_or_update_dev_table(schema: str, table: str, realtime_yaml: str, source_sql: str, dry_run: bool = False, apply: bool = False):
    db_log = get_db_logger()
    error_log = get_error_logger()
    
    cols = load_sink_contract(realtime_yaml)
    ddl = _emit_create(schema, table, cols)
    
    db_log.info("ddl_generation_completed", 
               schema=schema, 
               table=table, 
               columns_count=len(cols),
               dry_run=dry_run, 
               apply=apply)
    
    db_log.debug("ddl_preview", ddl=ddl)
    
    if apply and not dry_run:
        try:
            # Load database config with environment variable substitution
            dbconf = load_config_env("config/db.yaml")
            vc = VerticaClient(dbconf)
            
            db_log.info("applying_table_ddl", 
                       schema=schema, 
                       table=table)
            
            # First, create the table if it doesn't exist
            vc.execute_dev_sql(ddl)
            
            # Then, check for and add any missing columns
            missing_cols = _get_missing_columns(vc, schema, table, cols)
            if missing_cols:
                db_log.info("adding_missing_columns", 
                           table=f"{schema}.{table}", 
                           missing_columns=list(missing_cols.keys()),
                           missing_count=len(missing_cols))
                _add_missing_columns(vc, schema, table, missing_cols)
            else:
                db_log.info("table_schema_up_to_date", 
                           table=f"{schema}.{table}",
                           total_columns=len(cols))
            
            db_log.info("ddl_application_completed", 
                       schema=schema, 
                       table=table)
                       
        except Exception as e:
            error_log.error("ddl_application_failed",
                           schema=schema,
                           table=table,
                           error=str(e),
                           error_type=type(e).__name__)
            raise
    
    elif dry_run:
        db_log.info("dry_run_ddl_preview", 
                   schema=schema, 
                   table=table,
                   would_apply=apply)
