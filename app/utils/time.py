from __future__ import annotations
import datetime as dt
from zoneinfo import ZoneInfo

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def combine_move_ts(date_str: str, hour: int, tz: str) -> dt.datetime:
    d = dt.date.fromisoformat(str(date_str))
    return dt.datetime(d.year, d.month, d.day, int(hour), 0, 0, tzinfo=ZoneInfo(tz))
