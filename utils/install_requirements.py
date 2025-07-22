#!/usr/bin/env python3
"""
Install Required Packages for Model Training
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("📦 Installing required packages for model training...")
    
    packages = [
        "optuna",
        "xgboost", 
        "lightgbm",
        "catboost",
        "streamlit",
        "plotly",
        "PyYAML",
        "fastfm"  # For factorization machines (optional)
    ]
    
    success_count = 0
    
    for package in packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
            success_count += 1
        else:
            print(f"✗ Failed to install {package}")
    
    print(f"\n🎉 Installation complete! {success_count}/{len(packages)} packages installed.")
    
    if success_count < len(packages):
        print("\n⚠️ Some packages failed to install. You may need to install them manually:")
        print("pip install optuna xgboost lightgbm catboost streamlit plotly PyYAML")
    
    print("\nRun the test script to verify setup:")
    print("python utils/test_model_training_setup.py")

if __name__ == "__main__":
    main()
