#!/usr/bin/env python3
"""
Production Deployment Verification Script
=========================================

This script performs final verification that the system is production-ready
and provides a deployment checklist.
"""

import os
import sys
from pathlib import Path

print("🚀 PRODUCTION DEPLOYMENT VERIFICATION")
print("=" * 50)

def check_file_structure():
    """Verify all required files and directories exist."""
    print("📁 Checking file structure...")
    
    required_files = [
        'manage.sh',
        'health_check.sh', 
        'config.py',
        '.env',
        'requirements.txt',
        'README.md',
        'PRODUCTION_DEPLOYMENT.md'
    ]
    
    required_dirs = [
        'agents/',
        'config/',
        'utils/',
        'data/',
        'logs/',
        'models/'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required files and directories present")
    return True

def check_permissions():
    """Check file permissions are correct."""
    print("🔐 Checking file permissions...")
    
    executable_files = ['manage.sh', 'health_check.sh']
    for file in executable_files:
        if Path(file).exists():
            stat = Path(file).stat()
            if not (stat.st_mode & 0o111):  # Check if executable
                print(f"⚠️  {file} is not executable")
                return False
    
    if Path('.env').exists():
        stat = Path('.env').stat()
        # Check if readable by owner only (600)
        if (stat.st_mode & 0o777) != 0o600:
            print(f"⚠️  .env permissions should be 600 for security")
            return False
    
    print("✅ File permissions are correct")
    return True

def check_python_dependencies():
    """Check if all required Python packages are installed."""
    print("🐍 Checking Python dependencies...")
    
    try:
        import pandas
        import numpy
        import catboost
        import vertica_python
        import psutil
        import dotenv
        import sklearn
        import matplotlib
        import seaborn
        print("✅ All Python dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_configuration():
    """Check configuration is complete."""
    print("⚙️  Checking configuration...")
    
    try:
        from config import settings
        
        # Check critical settings
        critical_settings = [
            'VERTICA_HOST', 'VERTICA_PORT', 'VERTICA_USER', 
            'VERTICA_PASSWORD', 'VERTICA_DB', 'TABLE_NAME'
        ]
        
        missing_settings = []
        for setting in critical_settings:
            if not getattr(settings, setting, None):
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"❌ Missing configuration: {missing_settings}")
            print("Update your .env file with production values")
            return False
        
        print("✅ Configuration is complete")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_disk_space():
    """Check available disk space."""
    print("💾 Checking disk space...")
    
    import psutil
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    
    if free_gb < 5:
        print(f"❌ Insufficient disk space: {free_gb:.1f}GB free (minimum 5GB required)")
        return False
    elif free_gb < 10:
        print(f"⚠️  Low disk space: {free_gb:.1f}GB free (recommended 10GB+)")
    else:
        print(f"✅ Sufficient disk space: {free_gb:.1f}GB free")
    
    return True

def main():
    print("🚀 PRODUCTION DEPLOYMENT VERIFICATION")
    print("=" * 50)
    
    checks = [
        check_file_structure,
        check_permissions, 
        check_python_dependencies,
        check_configuration,
        check_disk_space
    ]
    
    all_passed = True
    for check in checks:
        try:
            result = check()
            if not result:
                all_passed = False
            print()
        except Exception as e:
            print(f"❌ Check failed: {e}")
            all_passed = False
            print()
    
    print("=" * 50)
    if all_passed:
        print("✅ SYSTEM READY FOR PRODUCTION DEPLOYMENT!")
        print()
        print("📋 DEPLOYMENT CHECKLIST:")
        print("□ Run health check: ./health_check.sh --full")
        print("□ Test training pipeline: ./manage.sh history")  
        print("□ Test prediction pipeline: ./manage.sh realtime")
        print("□ Set up cron jobs for automation")
        print("□ Configure monitoring and alerts")
        print("□ Set up backup procedures")
        print()
        print("📖 See PRODUCTION_DEPLOYMENT.md for detailed steps")
        return True
    else:
        print("❌ SYSTEM NOT READY FOR PRODUCTION")
        print("Please fix the issues above before deploying")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
