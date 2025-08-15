# ===============================
# File: utils/sql.py
# ===============================
from __future__ import annotations

from datetime import datetime


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _event_ts_expr(settings) -> str:
    """Return SQL expr building event_ts from MoveDate/MoveHour, handling VARCHAR MoveDate with format.
    If MOVE_DATE_IS_DATE is true, we skip TO_TIMESTAMP.
    """
    if getattr(settings, 'MOVE_DATE_IS_DATE', False):
        base_day = "DATE_TRUNC('day', MoveDate)"
    else:
        fmt = getattr(settings, 'MOVE_DATE_FORMAT', 'MM/DD/YYYY')
        base_day = f"DATE_TRUNC('day', TO_TIMESTAMP(MoveDate, '{fmt}'))"
    return f"TIMESTAMPADD(HOUR, CAST(MoveHour AS INT), {base_day})"


def build_select_sql(table: str, from_local: datetime, to_local: datetime, settings=None) -> str:
    from_lit = _fmt(from_local)
    to_lit = _fmt(to_local)
    event_ts = _event_ts_expr(settings)
    return f"""
WITH src AS (
  SELECT
    MoveDate,
    MoveHour,
    MoveType,
    TerminalID,
    Desig,
    TokenCount,
    ContainerCount,
    {event_ts} AS event_ts_filter
  FROM {table}
)
SELECT 
    MoveDate,
    MoveHour,
    MoveType,
    TerminalID,
    Desig,
    TokenCount,
    ContainerCount
FROM src
WHERE event_ts_filter >= TIMESTAMP '{from_lit}'
  AND event_ts_filter <  TIMESTAMP '{to_lit}'
ORDER BY event_ts_filter ASC;
""".strip()


def discover_max_event_ts(conn, table: str, settings=None):
    event_ts = _event_ts_expr(settings)
    sql = f"SELECT MAX({event_ts}) AS max_ts FROM {table};"
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return row[0]

# ===============================