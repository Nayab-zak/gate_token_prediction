# Production Configuration and Health Checks
"""
Production-specific configuration and health check utilities.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import psutil

from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

class ProductionValidator:
    """Validates system readiness for production deployment."""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
    
    def validate_environment(self) -> bool:
        """Validate production environment requirements."""
        logger.info("🔍 Starting production environment validation...")
        
        # Check critical environment variables
        critical_vars = [
            'VERTICA_HOST', 'VERTICA_PORT', 'VERTICA_USER', 
            'VERTICA_PASSWORD', 'VERTICA_DB', 'TABLE_NAME'
        ]
        
        for var in critical_vars:
            if not getattr(settings, var, None):
                self.errors.append(f"Missing critical environment variable: {var}")
        
        # Check disk space (minimum 5GB free)
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 5:
            self.errors.append(f"Insufficient disk space: {free_gb:.1f}GB free (minimum 5GB required)")
        elif free_gb < 10:
            self.warnings.append(f"Low disk space: {free_gb:.1f}GB free (recommended 10GB+)")
        
        # Check memory (minimum 2GB available)
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 2:
            self.errors.append(f"Insufficient memory: {available_gb:.1f}GB available (minimum 2GB required)")
        
        # Check Python packages
        try:
            import catboost
            import pandas
            import vertica_python
        except ImportError as e:
            self.errors.append(f"Missing required package: {e}")
        
        # Check model file exists
        model_path = Path(settings.MODEL_DIR) / "model.cbm"
        if not model_path.exists():
            self.errors.append(f"Model file not found: {model_path}")
        
        # Check database connectivity
        try:
            from agents.ingestion_agent import _connect
            with _connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            self.checks.append("Database connectivity: OK")
        except Exception as e:
            self.errors.append(f"Database connection failed: {e}")
        
        return len(self.errors) == 0
    
    def validate_data_freshness(self, max_age_hours: int = 24) -> bool:
        """Check if training data is reasonably fresh."""
        try:
            model_path = Path(settings.MODEL_DIR) / "model.cbm"
            if not model_path.exists():
                self.warnings.append("No trained model found")
                return False
                
            model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            if model_age > timedelta(days=7):
                self.warnings.append(f"Model is {model_age.days} days old - consider retraining")
            
            return True
        except Exception as e:
            self.warnings.append(f"Could not check model freshness: {e}")
            return False
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("🚀 PRODUCTION READINESS REPORT")
        print("="*60)
        
        if self.checks:
            print("\n✅ PASSED CHECKS:")
            for check in self.checks:
                print(f"  • {check}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.errors:
            print("\n❌ CRITICAL ISSUES:")
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n" + "="*60)
        if self.errors:
            print("❌ SYSTEM NOT READY FOR PRODUCTION")
            print("Please fix the critical issues above before deploying.")
        elif self.warnings:
            print("⚠️ SYSTEM READY WITH WARNINGS")
            print("Consider addressing the warnings for optimal performance.")
        else:
            print("✅ SYSTEM READY FOR PRODUCTION")
        print("="*60)

def check_production_readiness() -> bool:
    """Main function to check production readiness."""
    validator = ProductionValidator()
    is_ready = validator.validate_environment()
    validator.validate_data_freshness()
    validator.print_report()
    return is_ready and len(validator.warnings) == 0

if __name__ == "__main__":
    ready = check_production_readiness()
    sys.exit(0 if ready else 1)
