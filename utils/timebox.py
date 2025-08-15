# ===============================
# File: utils/timebox.py
# ===============================
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo('UTC'))


def to_local_naive(dt_utc: datetime, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    return dt_utc.astimezone(tz).replace(tzinfo=None)


def parse_human_ts(s: str, tz_name: str) -> datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' as local-naive datetime."""
    s = s.strip()
    tz = ZoneInfo(tz_name)
    if 'T' in s:
        # exact timestamp
        dt = datetime.fromisoformat(s)
    else:
        dt = datetime.fromisoformat(s + 'T00:00:00')
    return dt.replace(tzinfo=None)


def local_date_range(move_date_value: str | object) -> str:
    """Normalize MoveDate-like values to YYYY-MM-DD string for partition folder.
    Accepts 'YYYY-MM-DD', 'MM/DD/YYYY', or date-like objects.
    """
    if move_date_value is None:
        return 'unknown'
    s = str(move_date_value)
    try:
        if '/' in s:  # MM/DD/YYYY
            m, d, y = s.split('/')
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        # try ISO first
        return datetime.fromisoformat(s).date().isoformat()
    except Exception:
        return s.replace('/', '-').split(' ')[0]