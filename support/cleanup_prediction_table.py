#!/usr/bin/env python
"""
Script to drop and recreate the prediction table with new BI columns
This handles the SUR_GKEY IDENTITY column addition and other BI columns
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

def cleanup_prediction_table(conn):
    """Drop and prepare for recreation of prediction table with new BI columns"""
    
    schema = os.getenv("DEFAULT_SCHEMA", "DPW_DL")
    table = os.getenv("DEFAULT_TABLE", "T_DA_PRED_GATE_TOKEN")
    
    print(f"Using schema: {schema}, table: {table}")
    
    # First, check if table exists
    check_sql = f"""
    SELECT table_name 
    FROM tables 
    WHERE table_schema = '{schema}' 
    AND table_name = '{table}'
    """
    
    with conn.cursor() as cur:
        try:
            cur.execute(check_sql)
            result = cur.fetchall()
            if result:
                print(f"✓ Found table {schema}.{table}")
            else:
                print(f"⚠ Table {schema}.{table} does not exist")
                return
        except Exception as e:
            print(f"⚠ Could not check table existence: {e}")
    
    cleanup_sql = f"""
    DROP PROJECTION IF EXISTS {schema}.{table}_uniq CASCADE;
    DROP TABLE IF EXISTS {schema}.{table} CASCADE;
    """
    
    with conn.cursor() as cur:
        statements = [s.strip() for s in cleanup_sql.split(';') if s.strip() and not s.strip().startswith('--')]
        for stmt in statements:
            if stmt:  # Skip empty statements
                try:
                    print(f"Executing: {stmt}")
                    cur.execute(stmt)
                    # Get the number of affected objects
                    print("✓ Command executed successfully")
                    
                    # Check for any notices/warnings
                    if hasattr(cur, 'notices') and cur.notices:
                        for notice in cur.notices:
                            print(f"  Notice: {notice}")
                            
                except Exception as e:
                    print(f"❌ Error executing statement: {e}")
                    print(f"   Statement was: {stmt}")
                    # Don't raise, continue with other statements
    
    # Verify table is dropped
    with conn.cursor() as cur:
        try:
            cur.execute(check_sql)
            result = cur.fetchall()
            if not result:
                print(f"✓ Table {schema}.{table} successfully dropped")
            else:
                print(f"⚠ Table {schema}.{table} still exists!")
                print("Trying alternative drop method...")
                
                # Try more aggressive approach
                try:
                    # First try to drop all projections for this table
                    print("Dropping all projections for this table...")
                    drop_proj_sql = f"""
                    SELECT 'DROP PROJECTION IF EXISTS ' || projection_schema || '.' || projection_name || ' CASCADE;'
                    FROM projections 
                    WHERE anchor_table_name = '{table}' 
                    AND projection_schema = '{schema}'
                    """
                    cur.execute(drop_proj_sql)
                    proj_drops = cur.fetchall()
                    
                    for (drop_stmt,) in proj_drops:
                        print(f"Executing: {drop_stmt}")
                        cur.execute(drop_stmt)
                    
                    # Now try to drop the table again
                    final_drop = f"DROP TABLE {schema}.{table} CASCADE"
                    print(f"Executing: {final_drop}")
                    cur.execute(final_drop)
                    print("✓ Table dropped with alternative method")
                    
                except Exception as e2:
                    print(f"❌ Alternative method also failed: {e2}")
                    print("You may need to drop the table manually using a database client")
                    
        except Exception as e:
            print(f"⚠ Could not verify table deletion: {e}")
    
    print(f"Cleanup completed for {schema}.{table}!")
    print("Now you can run: python -m app.cli create-dev-table --apply")

def main():
    ap = argparse.ArgumentParser("Cleanup prediction table for schema update")
    ap.add_argument("--confirm", action="store_true", help="Confirm you want to drop the table")
    args = ap.parse_args()
    
    if not args.confirm:
        print("This will DROP the prediction table and all its data!")
        print("Make sure you have backups if needed.")
        print("Run with --confirm flag if you're sure:")
        print("python cleanup_prediction_table.py --confirm")
        sys.exit(1)
    
    load_dotenv()
    
    # Debug: Show what env vars we found
    print("Environment variables:")
    print(f"  DEFAULT_SCHEMA: {os.getenv('DEFAULT_SCHEMA', 'Not set')}")
    print(f"  DEFAULT_TABLE: {os.getenv('DEFAULT_TABLE', 'Not set')}")
    print(f"  VERTICA_DEV_HOST: {os.getenv('VERTICA_DEV_HOST', 'Not set')}")
    print(f"  VERTICA_DEV_DB: {os.getenv('VERTICA_DEV_DB', 'Not set')}")
    print()
    
    dev = _conn_from_env("VERTICA_DEV")
    try:
        print("Cleaning up prediction table with old schema...")
        cleanup_prediction_table(dev)
    finally:
        dev.close()

if __name__ == "__main__":
    main()
