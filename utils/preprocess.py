# ===============================
# File: utils/preprocess.py
# ===============================
from __future__ import annotations

import re
from typing import Iterable
from datetime import datetime

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------- I/O ----------

def load_csv_safely(path) -> pd.DataFrame:
    # Read as strings first; we will coerce types explicitly
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=['', 'NA', 'NaN', 'null', 'None'])


# ---------- Sanitization & Types ----------

def sanitize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def _normalize_date_format(s: pd.Series, fmt: str) -> pd.Series:
    """Normalize mixed date formats to a consistent format."""
    mapping = {'YYYY': '%Y', 'YY': '%y', 'MM': '%m', 'DD': '%d'}
    pyfmt = fmt
    for k, v in mapping.items():
        pyfmt = pyfmt.replace(k, v)
    
    # Try primary format first
    result = pd.to_datetime(s, format=pyfmt, errors='coerce')
    
    # For remaining NaT values, try alternate formats
    mask = result.isna()
    if mask.any():
        # Try the other common format
        alt_fmt = '%d/%m/%Y' if pyfmt == '%m/%d/%Y' else '%m/%d/%Y'
        result[mask] = pd.to_datetime(s[mask], format=alt_fmt, errors='coerce')
        
        # If still have NaT, try general parsing as last resort
        mask2 = result.isna()
        if mask2.any():
            result[mask2] = pd.to_datetime(s[mask2], errors='coerce')
    
    # Convert back to consistent string format (MM/DD/YYYY)
    return result.dt.strftime('%m/%d/%Y')


def coerce_types(df: pd.DataFrame, *, move_date_is_date: bool, move_date_fmt: str) -> pd.DataFrame:
    # MoveHour → int 0..23
    if 'MoveHour' in df.columns:
        df['MoveHour'] = pd.to_numeric(df['MoveHour'], errors='coerce').fillna(0).astype(int).clip(0, 23)
    
    # Normalize MoveDate to consistent format (handles mixed MM/DD/YYYY and DD/MM/YYYY)
    if 'MoveDate' in df.columns:
        df['MoveDate'] = _normalize_date_format(df['MoveDate'], move_date_fmt)
    
    # numeric counts
    for col in [c for c in df.columns if c.lower().endswith('count') or c.lower().endswith('qty')]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['MoveType', 'TerminalID', 'Desig']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    # Normalize common variants
    if 'MoveType' in df.columns:
        df['MoveType'] = df['MoveType'].replace({'INBOUND': 'IN', 'OUTBOUND': 'OUT'})
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = [c for c in ['MoveDate', 'MoveHour', 'MoveType', 'TerminalID', 'Desig'] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped:
        logger.info(f'Dropped {dropped} rows with nulls in {required}')
    return df


def dedupe_by_key(df: pd.DataFrame, *, key_cols: list[str]) -> pd.DataFrame:
    keys = [c for c in key_cols if c in df.columns]
    if not keys:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep='last')
    dup = before - len(df)
    if dup:
        logger.info(f'Removed {dup} duplicate rows by key={keys}')
    return df


# ---------- Outliers & QA ----------

def winsorize_numeric(df: pd.DataFrame, *, cols: list[str], iqr_k: float = 3.0) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        df[c] = s.clip(lower=lo, upper=hi)
    return df


def make_qa_report(df: pd.DataFrame, *, original_rows: int | None = None) -> dict:
    report = {
        'rows': int(len(df)),
        'date_min': str(df['MoveDate'].min()) if 'MoveDate' in df.columns and len(df) else None,
        'date_max': str(df['MoveDate'].max()) if 'MoveDate' in df.columns and len(df) else None,
        'null_counts': {c: int(df[c].isna().sum()) for c in df.columns},
    }
    if original_rows is not None:
        report['original_rows'] = int(original_rows)
        report['dropped_rows'] = int(original_rows - len(df))
    return report
