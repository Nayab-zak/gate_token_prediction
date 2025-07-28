#!/usr/bin/env python3
"""
Simple Vercel EDA Deployment Script - Step by step execution
"""

import os
import json
import pandas as pd
import yaml
from pathlib import Path
import shutil
from datetime import datetime

def main():
    print("🚀 Starting Vercel EDA Deployment...")
    project_dir = Path(".")
    vercel_dir = project_dir / "vercel-eda-app"
    
    # Step 1: Create directory structure
    print("🏗️ Creating directory structure...")
    if vercel_dir.exists():
        shutil.rmtree(vercel_dir)
    
    directories = [
        "pages", "components", "public/images", "public/data", "styles"
    ]
    
    for dir_path in directories:
        (vercel_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")
    
    # Step 2: Copy images
    print("📊 Copying visualization images...")
    image_files = []
    for img_file in project_dir.glob("model_analysis_*.png"):
        dest_path = vercel_dir / "public" / "images" / img_file.name
        shutil.copy2(img_file, dest_path)
        image_files.append(img_file.name)
        print(f"  📄 Copied {img_file.name}")
    
    # Step 3: Collect model data
    print("🤖 Collecting model data...")
    models_data = {"_champion": "random_forest"}  # Default
    
    # Try to read champion
    champion_file = project_dir / "models" / "champion.txt"
    if champion_file.exists():
        models_data["_champion"] = champion_file.read_text().strip()
    
    # Basic model info
    predictions_dir = project_dir / "data" / "predictions"
    if predictions_dir.exists():
        for model_dir in predictions_dir.iterdir():
            if model_dir.is_dir():
                models_data[model_dir.name] = {
                    "metadata": {"model": model_dir.name, "data_type": "dense"},
                    "total_predictions": 1000,  # Placeholder
                    "sample_predictions": []
                }
    
    # Save model data
    with open(vercel_dir / "public" / "data" / "models_data.json", "w") as f:
        json.dump(models_data, f, indent=2)
    
    print(f"✅ Collected data for {len(models_data)-1} models")
    print(f"📁 Vercel app created at: {vercel_dir}")
    print(f"🖼️ Images copied: {len(image_files)}")

if __name__ == "__main__":
    main()
