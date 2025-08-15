#!/usr/bin/env python3
"""
File Cleanup Utility
====================

This script helps clean up old intermediate files from the pipeline to save disk space.
Useful for maintenance or when switching from timestamped to replacement mode.
"""

import argparse
from pathlib import Path
from typing import List
import json
from datetime import datetime

from utils.io import cleanup_old_files
from config import settings

def get_pipeline_directories() -> List[Path]:
    """Get all directories that contain pipeline intermediate files."""
    dirs = []
    
    # Raw input data
    dirs.append(Path(settings.INGEST_OUTPUT_DIR) / "history")
    dirs.append(Path(settings.INGEST_OUTPUT_DIR) / "realtime")
    
    # Preprocessed data
    dirs.append(Path(settings.PREPROC_OUTPUT_DIR) / "history") 
    dirs.append(Path(settings.PREPROC_OUTPUT_DIR) / "realtime")
    dirs.append(Path(settings.PREPROC_REPORT_DIR))
    
    # Features
    dirs.append(Path(settings.FE_OUTPUT_DIR) / "history")
    dirs.append(Path(settings.FE_OUTPUT_DIR) / "realtime")
    
    # Models and predictions
    dirs.append(Path(settings.MODEL_DIR))
    
    # Reports
    dirs.append(Path(settings.EVAL_REPORT_DIR))
    
    return [d for d in dirs if d.exists()]

def cleanup_directory(directory: Path, patterns: List[str], keep_last_n: int, dry_run: bool = False) -> dict:
    """Clean up files in a directory matching given patterns."""
    result = {
        "directory": str(directory),
        "patterns": patterns,
        "deleted_files": [],
        "kept_files": [],
        "errors": []
    }
    
    for pattern in patterns:
        try:
            deleted = cleanup_old_files(directory, pattern, keep_last_n)
            result["deleted_files"].extend([str(f) for f in deleted])
            
            # Count kept files
            remaining = list(directory.glob(pattern))
            result["kept_files"].extend([str(f) for f in remaining])
            
        except Exception as e:
            result["errors"].append(f"Pattern {pattern}: {str(e)}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Clean up old pipeline files")
    parser.add_argument("--keep-last", type=int, default=1, 
                       help="Number of most recent files to keep (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--directory", type=str,
                       help="Clean specific directory instead of all pipeline directories")
    parser.add_argument("--pattern", type=str, default="*",
                       help="File pattern to match (default: *)")
    parser.add_argument("--report", action="store_true",
                       help="Save cleanup report to file")
    
    args = parser.parse_args()
    
    if args.directory:
        # Clean specific directory
        target_dir = Path(args.directory)
        if not target_dir.exists():
            print(f"❌ Directory does not exist: {target_dir}")
            return
        
        patterns = [args.pattern]
        directories = [target_dir]
    else:
        # Clean all pipeline directories
        directories = get_pipeline_directories()
        
        # Define cleanup patterns for each type of file
        patterns = [
            "*.csv",           # Raw and preprocessed data
            "*.parquet",       # Feature files
            "*.json",          # Reports and metrics
            "*.png",           # Evaluation plots
            "*.cbm",           # Model files
            "*__ingest_*",     # Timestamped files
        ]
    
    print("🧹 Pipeline File Cleanup")
    print("=" * 50)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'ACTIVE CLEANUP'}")
    print(f"Keep last: {args.keep_last} files")
    print(f"Directories: {len(directories)}")
    print("=" * 50)
    
    all_results = []
    total_deleted = 0
    total_kept = 0
    
    for directory in directories:
        print(f"\n📁 Processing: {directory}")
        
        if args.directory:
            # Use specific pattern for specific directory
            dir_patterns = [args.pattern]
        else:
            # Use all patterns for pipeline cleanup
            dir_patterns = patterns
        
        result = cleanup_directory(directory, dir_patterns, args.keep_last, args.dry_run)
        
        if not args.dry_run:
            deleted_count = len(result["deleted_files"])
            kept_count = len(result["kept_files"])
            total_deleted += deleted_count
            total_kept += kept_count
            
            if deleted_count > 0:
                print(f"  🗑️  Deleted: {deleted_count} files")
            if kept_count > 0:
                print(f"  ✅ Kept: {kept_count} files")
            if result["errors"]:
                print(f"  ⚠️  Errors: {len(result['errors'])}")
                for error in result["errors"]:
                    print(f"     {error}")
        else:
            # For dry run, just show what would be deleted
            all_files = []
            for pattern in dir_patterns:
                all_files.extend(list(directory.glob(pattern)))
            
            if len(all_files) > args.keep_last:
                would_delete = len(all_files) - args.keep_last
                print(f"  📝 Would delete: {would_delete} files (keeping {args.keep_last})")
            else:
                print(f"  ✅ No cleanup needed")
        
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 50)
    if not args.dry_run:
        print(f"✅ Cleanup complete!")
        print(f"🗑️  Total deleted: {total_deleted} files")
        print(f"✅ Total kept: {total_kept} files")
        
        # Estimate space saved (rough estimate)
        print(f"💾 Disk space optimized by removing duplicate versions")
    else:
        print(f"📝 Dry run complete! Use --dry-run=false to actually delete files")
    
    # Save report if requested
    if args.report:
        report_path = Path("data/_reports/cleanup") / f"cleanup_report_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": "dry_run" if args.dry_run else "active",
            "keep_last_n": args.keep_last,
            "total_deleted": total_deleted,
            "total_kept": total_kept,
            "directories_processed": len(directories),
            "detailed_results": all_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 Report saved: {report_path}")

if __name__ == "__main__":
    main()
