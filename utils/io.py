# ===============================
# File: utils/io.py
# ===============================
from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def timestamped_path(base_dir: Path, prefix: str, suffix: str = '') -> Path:
    ensure_dir(base_dir)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    return base_dir / f"{prefix}__ingest_{ts}{suffix}"

def simple_path(base_dir: Path, filename: str) -> Path:
    """Generate a simple non-timestamped path that will overwrite existing files."""
    ensure_dir(base_dir)
    return base_dir / filename

def cleanup_old_files(directory: Path, pattern: str, keep_last_n: int = 1) -> List[Path]:
    """
    Clean up old files matching a pattern, keeping only the last N versions.
    
    Args:
        directory: Directory to search in
        pattern: File pattern to match (e.g., "*.csv", "features_*.parquet")
        keep_last_n: Number of most recent files to keep
        
    Returns:
        List of files that were deleted
    """
    if not directory.exists():
        return []
    
    # Find all matching files
    matching_files = list(directory.glob(pattern))
    if len(matching_files) <= keep_last_n:
        return []
    
    # Sort by modification time (newest first)
    matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Delete older files beyond keep_last_n
    files_to_delete = matching_files[keep_last_n:]
    deleted_files = []
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_files.append(file_path)
        except (OSError, FileNotFoundError):
            # File might have been deleted by another process
            pass
    
    return deleted_files

def get_output_path(base_dir: Path, filename: str, replace_files: bool = True, 
                   keep_last_n: int = 1, cleanup_pattern: str = None) -> Path:
    """
    Get output path for a file, with optional cleanup of old versions.
    
    Args:
        base_dir: Base directory for the file
        filename: Target filename
        replace_files: If True, use simple filename; if False, use timestamped
        keep_last_n: Number of old versions to keep during cleanup
        cleanup_pattern: Pattern for cleanup (defaults to filename with wildcards)
        
    Returns:
        Path object for the output file
    """
    ensure_dir(base_dir)
    
    if replace_files:
        output_path = simple_path(base_dir, filename)
        
        # Clean up old files if pattern is provided
        if cleanup_pattern and keep_last_n < 999:  # 999 = effectively unlimited
            cleanup_old_files(base_dir, cleanup_pattern, keep_last_n)
            
        return output_path
    else:
        # Use timestamped path (legacy behavior)
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        suffix = f'.{ext}' if ext else ''
        return timestamped_path(base_dir, name, suffix)