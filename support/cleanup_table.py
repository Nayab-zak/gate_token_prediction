#!/usr/bin/env python
"""
Script to drop and recreate tables that need schema changes
This handles the COUNT_BATCH400 column removal
"""
import os, sys, argparse
from dotenv import load_dotenv

try:
    import vertica_python
except Exception:
    print("Missing dependency: pip install vertica-python python-dotenv")
    sys.exit(1)

def _conn_from_env(prefix: str):
    cfg = {
        "host": os.getenv(f"{prefix}_HOST"),
        "port": int(os.getenv(f"{prefix}_PORT", "5433")),
        "database": os.getenv(f"{prefix}_DB"),
        "user": os.getenv(f"{prefix}_USER"),
        "password": os.getenv(f"{prefix}_PASSWORD"),
        "autocommit": True,
        "connection_timeout": 10,
        "tlsmode": os.getenv(f"{prefix}_TLSMODE", "disable"),
    }
    missing = [k for k,v in cfg.items() if v in (None, "") and k not in ("connection_timeout","autocommit")]
    if missing:
        raise RuntimeError(f"Missing env vars for {prefix}: {missing}")
    return vertica_python.connect(**cfg)

def cleanup_and_recreate(conn):
    """Drop and recreate tables that need schema changes"""
    
    cleanup_sql = """
    -- Drop projections first (they depend on tables)
    DROP PROJECTION IF EXISTS DPW_DL.T_DA_CX_EXCEPTION_SUMMARY_PRJ CASCADE;
    DROP PROJECTION IF EXISTS DPW_DL.T_DA_CX_EXCEPTION_DETAILS_PRJ CASCADE;
    DROP PROJECTION IF EXISTS DPW_DL.T_DA_CX_EXCEPTION_RUNS_PRJ CASCADE;
    DROP PROJECTION IF EXISTS DPW_DL.T_DA_CX_EXCEPTION_RULES_PRJ CASCADE;
    
    -- Drop old DPW_DQ projections if they exist
    DROP PROJECTION IF EXISTS DPW_DQ.cm_summary_prj CASCADE;
    DROP PROJECTION IF EXISTS DPW_DQ.sh_summary_prj CASCADE;
    DROP PROJECTION IF EXISTS DPW_DQ.sh_exception_details_prj CASCADE;
    DROP PROJECTION IF EXISTS DPW_DQ.cm_exception_details_prj CASCADE;
    
    -- Truncate tables to clear data but keep structure temporarily
    TRUNCATE TABLE DPW_DL.T_DA_CX_EXCEPTION_SUMMARY;
    TRUNCATE TABLE DPW_DL.T_DA_CX_EXCEPTION_DETAILS;
    TRUNCATE TABLE DPW_DL.T_DA_CX_EXCEPTION_RUNS;
    TRUNCATE TABLE DPW_DL.T_DA_CX_EXCEPTION_RULES;
    
    -- Truncate DPW_DQ tables if they exist
    TRUNCATE TABLE DPW_DQ.cm_summary;
    TRUNCATE TABLE DPW_DQ.sh_summary;
    TRUNCATE TABLE DPW_DQ.sh_exception_details;
    TRUNCATE TABLE DPW_DQ.cm_exception_details;
    TRUNCATE TABLE DPW_DQ.dq_runs;
    TRUNCATE TABLE DPW_DQ.dq_rules;
    
    -- Now drop the tables with problematic schemas
    DROP TABLE IF EXISTS DPW_DL.T_DA_CX_EXCEPTION_SUMMARY CASCADE;
    DROP TABLE IF EXISTS DPW_DQ.cm_summary CASCADE;
    """
    
    with conn.cursor() as cur:
        statements = [s.strip() for s in cleanup_sql.split(';') if s.strip()]
        for stmt in statements:
            try:
                print(f"Executing: {stmt[:50]}...")
                cur.execute(stmt)
                print("✓ Success")
            except Exception as e:
                print(f"⚠ Warning (may be expected): {e}")
    
    print("Cleanup completed!")

def main():
    ap = argparse.ArgumentParser("Cleanup and recreate tables")
    ap.add_argument("--env", default="config.env")
    args = ap.parse_args()
    load_dotenv(args.env)
    
    dev = _conn_from_env("VERTICA_DEV")
    try:
        print("Cleaning up tables with old schema...")
        cleanup_and_recreate(dev)
        # print("\nNow run create_tables.py to recreate with correct schema:")
        # print("python create_tables.py --seed")
    finally:
        dev.close()

if __name__ == "__main__":
    main()
