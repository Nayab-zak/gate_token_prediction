# ===============================
# File: utils/dedupe.py
# ===============================
from __future__ import annotations

from typing import List, Dict, Any


def dedupe_rows(rows: List[Dict[str, Any]], key_columns: list[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple(r.get(c) for c in key_columns)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out