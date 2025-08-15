# ===============================
# File: agents/ingestion_agent.py
# ===============================
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Dict, Any

import vertica_python

from config import settings
from utils.timebox import now_utc, to_local_naive, parse_human_ts, local_date_range
from utils.sql import build_select_sql, discover_max_event_ts
from utils.io import ensure_dir, timestamped_path, get_output_path
from utils.dedupe import dedupe_rows
from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Bounds:
    from_local: datetime  # naive local time (TIMEZONE in settings)
    to_local: datetime    # naive local time (TIMEZONE in settings)


def _compute_bounds() -> Bounds:
    tz = settings.TIMEZONE
    if settings.INGEST_MODE == 'realtime':
        to_utc = now_utc()
        to_local = to_local_naive(to_utc, tz)
        from_local = to_local - timedelta(days=settings.WINDOW_DAYS) - timedelta(hours=settings.LOOKBACK_HOURS)
        return Bounds(from_local=from_local, to_local=to_local)

    if settings.INGEST_MODE == 'history':
        if not settings.HISTORY_START:
            from_local = parse_human_ts('1970-01-01', tz)
        else:
            from_local = parse_human_ts(settings.HISTORY_START, tz)

        if settings.HISTORY_END:
            to_local = parse_human_ts(settings.HISTORY_END, tz)
        else:
            with _connect() as conn:
                to_local = discover_max_event_ts(conn, settings.TABLE_NAME, settings)
        return Bounds(from_local=from_local, to_local=to_local)

    raise ValueError("INGEST_MODE must be 'realtime' or 'history'")


def _connect():
    info = {
        'host': settings.VERTICA_HOST,
        'port': settings.VERTICA_PORT,
        'user': settings.VERTICA_USER,
        'password': settings.VERTICA_PASSWORD,
        'database': settings.VERTICA_DB,
        'autocommit': True,
        'tlsmode': settings.VERTICA_TLSMODE,
    }
    return vertica_python.connect(**info)


def _row_batches(cur, batch_size: int) -> Iterable[List[tuple]]:
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def _tuples_to_dicts(columns: list[str], rows: List[tuple]) -> List[Dict[str, Any]]:
    return [dict(zip(columns, r)) for r in rows]


def _drop_column(rows: List[Dict[str, Any]], col: str | None) -> None:
    if not col:
        return
    for r in rows:
        r.pop(col, None)


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if not rows:
        out_path.touch(exist_ok=True)
        return
    header = list(rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def ingest_once() -> Path:
    bounds = _compute_bounds()
    logger.info(f"Mode={settings.INGEST_MODE} from={bounds.from_local} to={bounds.to_local} (local {settings.TIMEZONE})")

    sql = build_select_sql(
        table=settings.TABLE_NAME,
        from_local=bounds.from_local,
        to_local=bounds.to_local,
        settings=settings,
    )

    all_rows: List[Dict[str, Any]] = []

    with _connect() as conn:
        with conn.cursor() as cur:
            logger.info(f"Executing SQL: {sql}")
            cur.execute(sql)
            columns = [d[0] for d in cur.description]
            for chunk in _row_batches(cur, settings.BATCH_ROWS):
                dict_chunk = _tuples_to_dicts(columns, chunk)
                _drop_column(dict_chunk, settings.DROP_COL_NAME)
                dict_chunk = dedupe_rows(dict_chunk, key_columns=[c.strip() for c in settings.DEDUP_KEY.split(',')])
                all_rows.extend(dict_chunk)

    all_rows = dedupe_rows(all_rows, key_columns=[c.strip() for c in settings.DEDUP_KEY.split(',')])

    out_dir = Path(settings.INGEST_OUTPUT_DIR) / settings.INGEST_MODE

    if all_rows:
        table_name_clean = settings.TABLE_NAME.replace('.', '_')
        
        if settings.INGEST_MODE == 'history':
            # For history mode, create one consolidated CSV file
            filename = f'{table_name_clean}_history.csv'
            cleanup_pattern = f'{table_name_clean}_history*.csv'
        else:
            # For realtime mode, also create one consolidated CSV file  
            filename = f'{table_name_clean}_realtime.csv'
            cleanup_pattern = f'{table_name_clean}_realtime*.csv'
            
        out_path = get_output_path(
            base_dir=out_dir,
            filename=filename,
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=cleanup_pattern
        )
        _write_csv(all_rows, out_path)
        logger.info(f"Wrote: {out_path}")
        return out_path

    # Handle empty results - don't create empty files, raise informative error
    logger.warning(f"No data found in time window: {bounds.from_local} to {bounds.to_local}")
    logger.warning(f"Query returned 0 rows. Consider adjusting WINDOW_DAYS or time range.")
    raise ValueError(f"No data found for {settings.INGEST_MODE} mode in specified time window. "
                    f"From: {bounds.from_local}, To: {bounds.to_local}. "
                    f"Consider increasing WINDOW_DAYS (currently {settings.WINDOW_DAYS}) or "
                    f"check if data exists in the database for this period.")


if __name__ == '__main__':
    p = ingest_once()
    logger.info(f"Wrote: {p}")

# ===============================