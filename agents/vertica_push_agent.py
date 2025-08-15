from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import vertica_python

from config import settings
import os
from uuid import uuid4


@dataclass
class PushConfig:
    table: str
    create_table: bool
    upsert_mode: str
    batch_size: int
    include_true: bool


# ---------- Connection ----------

def _connect():
    info = {
        "host": settings.VERTICA_HOST,
        "port": settings.VERTICA_PORT,
        "user": settings.VERTICA_USER,
        "password": settings.VERTICA_PASSWORD,
        "database": settings.VERTICA_DB,
        "autocommit": True,
        "tlsmode": getattr(settings, "VERTICA_TLSMODE", "disable"),
    }
    return vertica_python.connect(**info)


# ---------- Table DDL ----------

def _ensure_table(conn, table: str, include_true: bool):
    """
    Creates the predictions table if not exists.
    Schema is conservative & explicit; adjust VARCHAR sizes if you know your limits.
    """
    cols_true = ", TokenCount_true FLOAT" if include_true else ""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        TerminalID       VARCHAR(64),
        MoveType         VARCHAR(16),
        Desig            VARCHAR(16),
        MoveDate_pred    DATE,
        MoveHour_pred    INT,
        TokenCount_pred  FLOAT,
        updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        {cols_true}
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


# ---------- Input handling ----------

_REQUIRED_COLS = [
    "TerminalID", "MoveType", "Desig",
    "MoveDate_pred", "MoveHour_pred", "TokenCount_pred"
]

def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    # Coerce types (defensive)
    df = df.copy()
    df["TerminalID"] = df["TerminalID"].astype(str)
    df["MoveType"] = df["MoveType"].astype(str)
    df["Desig"] = df["Desig"].astype(str)
    df["MoveHour_pred"] = pd.to_numeric(df["MoveHour_pred"], errors="coerce").astype("Int64")
    df["TokenCount_pred"] = pd.to_numeric(df["TokenCount_pred"], errors="coerce")

    # Date: try ISO first; fallback to common formats if needed
    try:
        df["MoveDate_pred"] = pd.to_datetime(df["MoveDate_pred"], errors="raise").dt.date
    except Exception:
        df["MoveDate_pred"] = pd.to_datetime(
            df["MoveDate_pred"], format="%Y-%m-%d", errors="coerce"
        ).dt.date

    # Drop rows with null key fields
    key_cols = ["TerminalID", "MoveType", "Desig", "MoveDate_pred", "MoveHour_pred"]
    df = df.dropna(subset=key_cols + ["TokenCount_pred"])
    df["MoveHour_pred"] = df["MoveHour_pred"].astype(int)

    # Optional: keep ground truth if present & configured
    if settings.PUSH_INCLUDE_TRUE and "TokenCount_true" in df.columns:
        df["TokenCount_true"] = pd.to_numeric(df["TokenCount_true"], errors="coerce")
    else:
        if "TokenCount_true" in df.columns:
            df = df.drop(columns=["TokenCount_true"])

    return df


# ---------- Batching ----------

def _chunks(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    n = len(df)
    for i in range(0, n, batch_size):
        yield df.iloc[i : i + batch_size]


# ---------- Upsert via MERGE (VALUES) ----------

def _merge_values_sql(table: str, include_true: bool, rows_count: int) -> Tuple[str, List[str]]:
    """
    Build a parameterized MERGE USING (VALUES ...) statement with rows_count tuples.
    Returns (sql, columns_order).
    """
    cols = [
        "TerminalID", "MoveType", "Desig", "MoveDate_pred", "MoveHour_pred", "TokenCount_pred"
    ]
    if include_true:
        cols.append("TokenCount_true")

    cols_str = ", ".join(cols)
    placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
    values_block = ",\n        ".join([placeholders] * rows_count)

    # ON key uses the first five columns (TerminalID, MoveType, Desig, MoveDate_pred, MoveHour_pred)
    on_clause = (
        "t.TerminalID = s.TerminalID AND "
        "t.MoveType = s.MoveType AND "
        "t.Desig = s.Desig AND "
        "t.MoveDate_pred = s.MoveDate_pred AND "
        "t.MoveHour_pred = s.MoveHour_pred"
    )

    set_list = ["TokenCount_pred = s.TokenCount_pred", "updated_at = CURRENT_TIMESTAMP"]
    if include_true:
        set_list.append("TokenCount_true = s.TokenCount_true")
    set_clause = ", ".join(set_list)

    insert_cols = [
        "TerminalID", "MoveType", "Desig", "MoveDate_pred", "MoveHour_pred", "TokenCount_pred", "updated_at"
    ]
    insert_vals = [
        "s.TerminalID", "s.MoveType", "s.Desig", "s.MoveDate_pred", "s.MoveHour_pred", "s.TokenCount_pred", "CURRENT_TIMESTAMP"
    ]
    if include_true:
        insert_cols.insert(6, "TokenCount_true")
        insert_vals.insert(6, "s.TokenCount_true")

    sql = f"""
MERGE INTO {table} AS t
USING (
    SELECT {cols_str} FROM (VALUES
        {values_block}
    ) AS v({cols_str})
) AS s
ON {on_clause}
WHEN MATCHED THEN UPDATE SET {set_clause}
WHEN NOT MATCHED THEN INSERT ({", ".join(insert_cols)})
VALUES ({", ".join(insert_vals)});
""".strip()
    return sql, cols


def _create_temp_and_load(conn, temp_name: str, batch: pd.DataFrame, include_true: bool):
    cols = [
        "TerminalID VARCHAR(64)",
        "MoveType VARCHAR(16)",
        "Desig VARCHAR(16)",
        "MoveDate_pred DATE",
        "MoveHour_pred INT",
        "TokenCount_pred FLOAT"
    ]
    if include_true:
        cols.append("TokenCount_true FLOAT")

    ddl = f"CREATE LOCAL TEMP TABLE {temp_name} ({', '.join(cols)}) ON COMMIT PRESERVE ROWS;"
    with conn.cursor() as cur:
        cur.execute(ddl)

        insert_cols = ["TerminalID","MoveType","Desig","MoveDate_pred","MoveHour_pred","TokenCount_pred"]
        if include_true:
            insert_cols.append("TokenCount_true")
        col_list = ", ".join(insert_cols)
        placeholders = "(" + ", ".join(["%s"] * len(insert_cols)) + ")"
        sql = f"INSERT INTO {temp_name} ({col_list}) VALUES {placeholders}"

        params = []
        for _, r in batch.iterrows():
            row = [
                str(r["TerminalID"]),
                str(r["MoveType"]),
                str(r["Desig"]),
                r["MoveDate_pred"],            # python date is fine
                int(r["MoveHour_pred"]),
                float(r["TokenCount_pred"]),
            ]
            if include_true:
                val_true = r.get("TokenCount_true")
                row.append(None if pd.isna(val_true) else float(val_true))
            params.append(tuple(row))
        cur.executemany(sql, params)


def _merge_upsert(conn, table: str, batch: pd.DataFrame, include_true: bool):
    temp = f"tmp_pred_{os.getpid()}_{uuid4().hex[:8]}"
    try:
        _create_temp_and_load(conn, temp, batch, include_true)
        set_list = ["TokenCount_pred = s.TokenCount_pred", "updated_at = CURRENT_TIMESTAMP"]
        if include_true:
            set_list.append("TokenCount_true = s.TokenCount_true")
        set_clause = ", ".join(set_list)

        insert_cols = ["TerminalID","MoveType","Desig","MoveDate_pred","MoveHour_pred","TokenCount_pred","updated_at"]
        insert_vals = ["s.TerminalID","s.MoveType","s.Desig","s.MoveDate_pred","s.MoveHour_pred","s.TokenCount_pred","CURRENT_TIMESTAMP"]
        if include_true:
            insert_cols.insert(6, "TokenCount_true")
            insert_vals.insert(6, "s.TokenCount_true")

        sql = f"""
MERGE INTO {table} t
USING {temp} s
ON (
  t.TerminalID = s.TerminalID AND
  t.MoveType   = s.MoveType   AND
  t.Desig      = s.Desig      AND
  t.MoveDate_pred = s.MoveDate_pred AND
  t.MoveHour_pred = s.MoveHour_pred
)
WHEN MATCHED THEN UPDATE SET {set_clause}
WHEN NOT MATCHED THEN INSERT ({", ".join(insert_cols)})
VALUES ({", ".join(insert_vals)});
""".strip()
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {temp};")


def _insert_only(conn, table: str, batch: pd.DataFrame, include_true: bool):
    cols = ["TerminalID","MoveType","Desig","MoveDate_pred","MoveHour_pred","TokenCount_pred"]
    if include_true:
        cols.append("TokenCount_true")
    cols.append("updated_at")

    col_list = ", ".join(cols)
    placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
    sql = f"INSERT INTO {table} ({col_list}) VALUES {placeholders}"

    params = []
    for _, r in batch.iterrows():
        row = [
            str(r["TerminalID"]),
            str(r["MoveType"]),
            str(r["Desig"]),
            r["MoveDate_pred"],
            int(r["MoveHour_pred"]),
            float(r["TokenCount_pred"]),
        ]
        if include_true:
            val_true = r.get("TokenCount_true")
            row.append(None if pd.isna(val_true) else float(val_true))
        row.append(None)  # updated_at -> let DB default or NULL; if you want CURRENT_TIMESTAMP, keep MERGE path
        params.append(tuple(row))

    with conn.cursor() as cur:
        cur.executemany(sql, params)

# ---------- Public API ----------

def push_predictions(
    preds_path: Path,
    cfg: PushConfig | None = None,
) -> int:
    """
    Push predictions CSV/Parquet to Vertica.
    Returns number of rows processed.
    """
    cfg = cfg or PushConfig(
        table=settings.PUSH_TABLE_NAME,
        create_table=settings.PUSH_CREATE_TABLE,
        upsert_mode=settings.PUSH_UPSERT_MODE,
        batch_size=settings.PUSH_BATCH_SIZE,
        include_true=settings.PUSH_INCLUDE_TRUE,
    )

    df = _load_predictions(preds_path)
    if df.empty:
        print("No predictions to push (empty file).")
        return 0

    with _connect() as conn:
        if cfg.create_table:
            _ensure_table(conn, cfg.table, cfg.include_true)

        total = 0
        for batch in _chunks(df, cfg.batch_size):
            if cfg.upsert_mode == "merge_values":
                _merge_upsert(conn, cfg.table, batch, cfg.include_true)
            elif cfg.upsert_mode == "insert_only":
                _insert_only(conn, cfg.table, batch, cfg.include_true)
            else:
                raise ValueError("PUSH_UPSERT_MODE must be 'merge_values' or 'insert_only'")
            total += len(batch)

    print(f"✅ Pushed {total} rows to {cfg.table} (mode={cfg.upsert_mode})")
    return total


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=settings.PREDICTIONS_PATH, help="Path to predictions CSV/Parquet")
    ap.add_argument("--table", default=settings.PUSH_TABLE_NAME, help="Destination table")
    ap.add_argument("--mode", default=settings.PUSH_UPSERT_MODE, choices=["merge_values","insert_only"])
    ap.add_argument("--no-create", action="store_true", help="Do not auto-create table")
    ap.add_argument("--batch", type=int, default=settings.PUSH_BATCH_SIZE)
    ap.add_argument("--no-true", action="store_true", help="Ignore TokenCount_true even if present")
    args = ap.parse_args()

    cfg = PushConfig(
        table=args.table,
        create_table=not args.no_create,
        upsert_mode=args.mode,
        batch_size=args.batch,
        include_true=not args.no_true and settings.PUSH_INCLUDE_TRUE,
    )
    push_predictions(Path(args.preds), cfg)


if __name__ == "__main__":
    main()
