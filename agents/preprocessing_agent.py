# ===============================
# File: agents/preprocessing_agent.py
# ===============================
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from config import settings
from utils.logging import get_logger
from utils.io import ensure_dir, timestamped_path, get_output_path
from utils.preprocess import (
    load_csv_safely,
    coerce_types,
    sanitize_strings,
    standardize_categories,
    drop_invalid_rows,
    dedupe_by_key,
    winsorize_numeric,
    make_qa_report,
)

logger = get_logger(__name__)

@dataclass
class PreprocConfig:
    input_dir: Path
    output_dir: Path
    mode: str
    output_format: str
    max_files: int | None


def _discover_inputs(input_dir: Path, max_files: int | None) -> List[Path]:
    files = list(input_dir.rglob('*.csv'))
    
    # Filter out empty files to avoid processing issues
    non_empty_files = []
    for f in files:
        if f.stat().st_size > 0:
            non_empty_files.append(f)
        else:
            logger.warning(f"Skipping empty file: {f}")
    
    # Sort by modification time (most recent first) instead of alphabetically
    files_sorted = sorted(non_empty_files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    return files_sorted[:max_files] if max_files else files_sorted


def _write(df: pd.DataFrame, base_dir: Path, mode: str, out_format: str) -> Path:
    ensure_dir(base_dir)
    
    if settings.PREPROC_PARTITION_BY_DATE and not settings.REPLACE_INTERMEDIATE_FILES:
        # Original logic: partition by date using MoveDate (only if not replacing files)
        if 'MoveDate' in df.columns:
            # Convert MoveDate to standard date format for partitioning
            df_temp = df.copy()
            if settings.MOVE_DATE_IS_DATE:
                dt_values = pd.to_datetime(df_temp['MoveDate']).dt.date.astype(str)
            else:
                # Parse MoveDate using the specified format
                fmt_map = {'MM/DD/YYYY': '%m/%d/%Y', 'DD/MM/YYYY': '%d/%m/%Y', 'YYYY-MM-DD': '%Y-%m-%d'}
                py_fmt = fmt_map.get(settings.MOVE_DATE_FORMAT, '%m/%d/%Y')
                dt_values = pd.to_datetime(df_temp['MoveDate'], format=py_fmt, errors='coerce').dt.date.astype(str)
        else:
            dt_values = pd.Series(['unknown'] * len(df))

        last_path: Path | None = None
        for day, part in df.groupby(dt_values):
            out_dir = base_dir / mode / f'dt={day}'
            ensure_dir(out_dir)
            out_path = timestamped_path(out_dir, prefix=settings.TABLE_NAME.replace('.', '_'), suffix=(f'.{out_format}'))
            if out_format == 'parquet':
                part.to_parquet(out_path, index=False)
            else:
                part.to_csv(out_path, index=False)
            last_path = out_path
        assert last_path is not None
        return last_path
    else:
        # New logic: create single consolidated file
        out_dir = base_dir / mode
        table_name_clean = settings.TABLE_NAME.replace('.', '_')
        filename = f'{table_name_clean}_preprocessed.{out_format}'
        cleanup_pattern = f'{table_name_clean}_*preprocessed*.{out_format}' if settings.REPLACE_INTERMEDIATE_FILES else None
        
        out_path = get_output_path(
            base_dir=out_dir,
            filename=filename,
            replace_files=settings.REPLACE_INTERMEDIATE_FILES,
            keep_last_n=settings.KEEP_LAST_N_VERSIONS,
            cleanup_pattern=cleanup_pattern
        )
        
        if out_format == 'parquet':
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return out_path


def _write_report(report: dict, reports_dir: Path) -> Path:
    ensure_dir(reports_dir)
    filename = 'preprocessing_report.json'
    cleanup_pattern = 'preprocessing_report*.json' if settings.REPLACE_INTERMEDIATE_FILES else None
    
    out_path = get_output_path(
        base_dir=reports_dir,
        filename=filename,
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern=cleanup_pattern
    )
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    return out_path


def run_preprocessing() -> Path:
    # Dynamically determine input directory based on INGEST_MODE
    input_base_dir = Path(settings.INGEST_OUTPUT_DIR) / settings.INGEST_MODE
    
    cfg = PreprocConfig(
        input_dir=input_base_dir,
        output_dir=Path(settings.PREPROC_OUTPUT_DIR),
        mode=settings.INGEST_MODE,
        output_format=settings.PREPROC_OUTPUT_FORMAT,
        max_files=(settings.PREPROC_MAX_FILES if settings.PREPROC_MAX_FILES > 0 else None),
    )

    inputs = _discover_inputs(cfg.input_dir, cfg.max_files)
    if not inputs:
        raise FileNotFoundError(f'No input CSV files found under {cfg.input_dir}')

    all_reports = []
    combined: list[pd.DataFrame] = []

    for path in inputs:
        logger.info(f'Preprocessing: {path}')
        df = load_csv_safely(path)
        original_rows = len(df)

        df = sanitize_strings(df)
        df = coerce_types(df, move_date_is_date=settings.MOVE_DATE_IS_DATE, move_date_fmt=settings.MOVE_DATE_FORMAT)
        df = standardize_categories(df)
        df = drop_invalid_rows(df)
        df = dedupe_by_key(df, key_cols=[c.strip() for c in settings.DEDUP_KEY.split(',')])

        if settings.PREPROC_WINSORIZE:
            df = winsorize_numeric(df, cols=[c for c in df.columns if c.lower().endswith('count')])

        report = make_qa_report(df, original_rows=original_rows)
        all_reports.append(report)
        combined.append(df)

    out_df = pd.concat(combined, ignore_index=True) if len(combined) > 1 else combined[0]

    out_path = _write(out_df, cfg.output_dir, cfg.mode, cfg.output_format)
    rep_path = _write_report({'files': len(inputs), 'per_file': all_reports, 'summary': make_qa_report(out_df)}, Path(settings.PREPROC_REPORT_DIR))

    logger.info(f'Wrote cleaned data → {out_path}')
    logger.info(f'Wrote QA report → {rep_path}')
    return out_path


if __name__ == '__main__':
    run_preprocessing()
