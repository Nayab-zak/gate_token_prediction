from __future__ import annotations
import polars as pl
from typing import Sequence

class ValidationError(Exception): ...

def assert_required_columns(df: pl.DataFrame, required: Sequence[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}; present={df.columns}")

def ensure_sorted(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return df.sort(keys)
