#!/usr/bin/env python3
"""
Toggle between development and production epoch settings.

Usage:
    python toggle_epochs.py [dev|prod]

This script modifies config.py to set epochs/iterations to development mode (low values)
or production mode (full training values).
"""

import os
import sys
import re

CONFIG_FILE = "config.py"
DEV_MODE_VALUE = 1  # Low epoch value for quick testing
PRODUCTION_MODE_VALUES = {
    "LSTM_CLASSIC_EPOCHS": 50,
    "LSTM_AUGMENTED_EPOCHS": 50,
    "MLP_CLASSIC_EPOCHS": 200,
    "MLP_AUGMENTED_EPOCHS": 200,
    "XGB_CLASSIC_ITERATIONS": 300,
    "XGB_AUGMENTED_ITERATIONS": 300,
    "LGBM_CLASSIC_ITERATIONS": 300,
    "LGBM_AUGMENTED_ITERATIONS": 300,
    "CATBOOST_CLASSIC_ITERATIONS": 500,
    "CATBOOST_AUGMENTED_ITERATIONS": 500,
    "RF_CLASSIC_ESTIMATORS": 100,
    "RF_AUGMENTED_ESTIMATORS": 100,
    "AE_MAX_EPOCHS": 100
}

EPOCH_PARAMETER_NAMES = list(PRODUCTION_MODE_VALUES.keys())


def read_config():
    """Read the current config.py file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {CONFIG_FILE}: {e}")
        sys.exit(1)


def write_config(content):
    """Write updated content to config.py"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"Error writing to {CONFIG_FILE}: {e}")
        sys.exit(1)


def set_dev_mode():
    """Set all epoch parameters to development mode (low values)"""
    print(f"Setting {CONFIG_FILE} to DEVELOPMENT mode (epochs = {DEV_MODE_VALUE})...")
    
    config_content = read_config()
    
    # Replace each parameter with development value
    for param in EPOCH_PARAMETER_NAMES:
        pattern = rf"({param}\s*=\s*)\d+"
        replacement = f"\\g<1>{DEV_MODE_VALUE}"
        config_content = re.sub(pattern, replacement, config_content)
    
    write_config(config_content)
    print("✅ Development mode activated - all epochs/iterations set to 1")


def set_prod_mode():
    """Set all epoch parameters to production mode (full values)"""
    print(f"Setting {CONFIG_FILE} to PRODUCTION mode (full training)...")
    
    config_content = read_config()
    
    # Replace each parameter with production value
    for param, value in PRODUCTION_MODE_VALUES.items():
        pattern = rf"({param}\s*=\s*)\d+"
        replacement = f"\\g<1>{value}"
        config_content = re.sub(pattern, replacement, config_content)
    
    write_config(config_content)
    print("✅ Production mode activated - all epochs/iterations set to full values")


def show_current_settings():
    """Display the current epoch settings"""
    config_content = read_config()
    print("\nCurrent epoch settings:")
    print("-" * 40)
    
    for param in EPOCH_PARAMETER_NAMES:
        pattern = rf"{param}\s*=\s*(\d+)"
        match = re.search(pattern, config_content)
        if match:
            value = match.group(1)
            print(f"{param:<30} = {value}")
    
    print("-" * 40)


def main():
    """Main function to handle command-line arguments"""
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ['dev', 'prod', 'status']:
        print("Usage: python toggle_epochs.py [dev|prod|status]")
        print("  dev    : Set all epochs to minimal values for development/testing")
        print("  prod   : Set all epochs to full values for production training")
        print("  status : Show current epoch settings")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == 'dev':
        set_dev_mode()
    elif mode == 'prod':
        set_prod_mode()
    elif mode == 'status':
        pass  # Just show current settings
    
    show_current_settings()


if __name__ == "__main__":
    main()
