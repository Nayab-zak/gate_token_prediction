#!/usr/bin/env python3
"""
Enhanced Pipeline Execution Script
=================================

This script executes the enhanced temporal validation pipeline and provides
comprehensive status updates and error handling.

EXECUTION STAGES:
1. ✅ Environment validation
2. ✅ Enhanced training (12 models)
3. ✅ Enhanced testing
4. ✅ Champion selection
5. ✅ Production monitoring setup
6. ✅ Deployment readiness assessment

Author: AI Assistant
Date: 2025-07-31
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

def print_banner(title: str, width: int = 80):
    """Print a formatted banner."""
    print("=" * width)
    print(f" {title} ".center(width))
    print("=" * width)

def print_status(status: str, message: str):
    """Print formatted status message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {status} {message}")

def check_environment():
    """Check if the environment is ready for enhanced pipeline execution."""
    print_banner("ENVIRONMENT VALIDATION")
    
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    
    # Check base directory
    if not os.path.exists(base_path):
        print_status("❌", f"Base directory not found: {base_path}")
        return False
    
    print_status("✅", f"Base directory found: {base_path}")
    
    # Check required directories
    required_dirs = [
        "agents", "utils", "data", "models"
    ]
    
    for dir_name in required_dirs:
        dir_path = f"{base_path}/{dir_name}"
        if os.path.exists(dir_path):
            print_status("✅", f"Directory exists: {dir_name}")
        else:
            print_status("❌", f"Directory missing: {dir_name}")
            return False
    
    # Check for enhanced agents
    enhanced_agents = [
        "agents/09_enhanced_training_agent_updater.py",
        "agents/10_enhanced_test_agent_updater.py", 
        "agents/11_production_monitoring_agent.py",
        "agents/12_enhanced_pipeline_orchestrator.py"
    ]
    
    for agent_path in enhanced_agents:
        full_path = f"{base_path}/{agent_path}"
        if os.path.exists(full_path):
            print_status("✅", f"Enhanced agent found: {agent_path}")
        else:
            print_status("❌", f"Enhanced agent missing: {agent_path}")
            return False
    
    # Check temporal validation utilities
    temporal_validation_path = f"{base_path}/utils/temporal_validation.py"
    if os.path.exists(temporal_validation_path):
        print_status("✅", "Temporal validation framework found")
    else:
        print_status("❌", "Temporal validation framework missing")
        return False
    
    print_status("🎉", "Environment validation completed successfully!")
    return True

def run_enhanced_pipeline():
    """Run the enhanced pipeline with comprehensive monitoring."""
    print_banner("ENHANCED PIPELINE EXECUTION")
    
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    
    # Change to project directory
    os.chdir(base_path)
    print_status("📁", f"Changed to directory: {base_path}")
    
    # Set Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{base_path}:{base_path}/agents:{base_path}/utils"
    
    try:
        # Run the enhanced pipeline orchestrator
        print_status("🚀", "Starting Enhanced Pipeline Orchestrator...")
        
        start_time = time.time()
        
        # Execute the orchestrator
        result = subprocess.run([
            sys.executable, 
            "agents/12_enhanced_pipeline_orchestrator.py"
        ], 
        env=env,
        capture_output=True, 
        text=True,
        timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check execution results
        if result.returncode == 0:
            print_status("✅", f"Pipeline completed successfully in {duration:.1f} seconds")
            print("\n" + "="*60)
            print("PIPELINE OUTPUT:")
            print("="*60)
            print(result.stdout)
            
            if result.stderr:
                print("\n" + "="*60)
                print("WARNINGS/INFO:")
                print("="*60)
                print(result.stderr)
            
            return True
        else:
            print_status("❌", f"Pipeline failed after {duration:.1f} seconds")
            print("\n" + "="*60)
            print("ERROR OUTPUT:")
            print("="*60)
            print(result.stderr)
            print("\nSTDOUT:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print_status("⏰", "Pipeline execution timed out after 1 hour")
        return False
    except Exception as e:
        print_status("❌", f"Pipeline execution failed: {str(e)}")
        return False

def run_individual_training_test():
    """Run a quick test of individual components."""
    print_banner("INDIVIDUAL COMPONENT TEST")
    
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    os.chdir(base_path)
    
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{base_path}:{base_path}/agents:{base_path}/utils"
    
    try:
        # Test temporal validation framework
        print_status("🧪", "Testing temporal validation framework...")
        
        test_script = """
import sys
sys.path.append('utils')
from temporal_validation import setup_temporal_logger, validate_temporal_inputs
import numpy as np
import pandas as pd

# Quick test
logger = setup_temporal_logger()
logger.info('Testing enhanced temporal validation framework...')

# Create sample data
n_samples = 50
timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')
X = np.random.randn(n_samples, 3)
y = np.random.randn(n_samples)

# Test input validation
X_val, y_val, ts_val = validate_temporal_inputs(X, y, timestamps, logger)
logger.info(f'Test completed successfully - shapes: X={X_val.shape}, y={y_val.shape}')
print('✅ Temporal validation framework test passed')
"""
        
        result = subprocess.run([
            sys.executable, "-c", test_script
        ], 
        env=env,
        capture_output=True, 
        text=True,
        timeout=30
        )
        
        if result.returncode == 0:
            print_status("✅", "Temporal validation framework test passed")
            print(result.stdout)
        else:
            print_status("❌", "Temporal validation framework test failed")
            print(result.stderr)
            return False
        
        return True
        
    except Exception as e:
        print_status("❌", f"Component test failed: {str(e)}")
        return False

def check_data_availability():
    """Check if required data files are available."""
    print_banner("DATA AVAILABILITY CHECK")
    
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    
    # Check for training data
    data_paths = [
        "data/features/train_features.csv",
        "data/features/test_features.csv",
        "data/encoded_output/train_encoded.csv",
        "data/encoded_output/test_encoded.csv"
    ]
    
    available_data = []
    
    for data_path in data_paths:
        full_path = f"{base_path}/{data_path}"
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            print_status("✅", f"Data found: {data_path} ({file_size:,} bytes)")
            available_data.append(data_path)
        else:
            print_status("⚠️", f"Data missing: {data_path}")
    
    if len(available_data) >= 2:  # At least train and test data
        print_status("✅", f"Sufficient data available ({len(available_data)}/4 files found)")
        return True
    else:
        print_status("❌", f"Insufficient data available ({len(available_data)}/4 files found)")
        print_status("💡", "Please run data preprocessing agents first")
        return False

def main():
    """Main execution function."""
    print_banner("ENHANCED TEMPORAL VALIDATION PIPELINE EXECUTOR", 100)
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Environment validation
    if not check_environment():
        print_status("❌", "Environment validation failed - cannot proceed")
        return False
    
    print()
    
    # Step 2: Data availability check
    if not check_data_availability():
        print_status("❌", "Data availability check failed - cannot proceed")
        return False
    
    print()
    
    # Step 3: Component test
    if not run_individual_training_test():
        print_status("❌", "Component test failed - cannot proceed")
        return False
    
    print()
    
    # Step 4: Run enhanced pipeline
    success = run_enhanced_pipeline()
    
    print()
    print_banner("EXECUTION SUMMARY")
    
    if success:
        print_status("🎉", "Enhanced pipeline executed successfully!")
        print_status("📊", "Check the pipeline_results directory for detailed reports")
        print_status("📁", "Logs are available in the logs directory")
        print_status("🔍", "Monitoring data is in the monitoring directory")
        
        # Print next steps
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Review deployment readiness report in pipeline_results/")
        print("2. If approved, deploy champion model to production")
        print("3. Set up continuous monitoring using the monitoring agent")
        print("4. Configure alerts and dashboards in Grafana")
        
    else:
        print_status("❌", "Enhanced pipeline execution failed")
        print_status("🔍", "Check error messages above for troubleshooting")
        print_status("📋", "Review logs in the logs directory for detailed error information")
    
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
