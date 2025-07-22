#!/usr/bin/env python3
"""
Test the system information functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the notebook equivalent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

def get_champion_model_info():
    """Get current champion model information"""
    champion_file = Path("models/champion.txt")
    if champion_file.exists():
        with open(champion_file, 'r') as f:
            champion_model = f.read().strip()
        return champion_model
    return "Unknown"

def load_model_metadata(model_name, predictions_dir="data/predictions"):
    """Load model metadata from YAML file"""
    model_dir = Path(predictions_dir) / model_name
    metadata_files = list(model_dir.glob("*_metadata_*.yaml"))
    
    if not metadata_files:
        return None
    
    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_metadata, 'r') as f:
            content = f.read()
            # Filter out problematic NumPy serializations
            lines = content.split('\n')
            filtered_lines = []
            skip_lines = False
            
            for line in lines:
                if 'test_metrics:' in line or 'train_metrics:' in line:
                    skip_lines = True
                    continue
                if skip_lines and (line.startswith('  ') or line.strip() == ''):
                    continue
                if skip_lines and not line.startswith(' '):
                    skip_lines = False
                
                if not skip_lines:
                    filtered_lines.append(line)
            
            filtered_content = '\n'.join(filtered_lines)
            metadata = yaml.safe_load(filtered_content)
            
            # Ensure we have the model name
            if metadata and 'model' not in metadata:
                metadata['model'] = model_name
        return metadata
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return {'model': model_name}

def test_system_info():
    """Test system information functionality"""
    print("🧪 TESTING SYSTEM INFORMATION FUNCTIONALITY")
    print("=" * 60)
    
    # Test champion model info
    champion = get_champion_model_info()
    print(f"✅ Champion model: {champion}")
    
    # Test metadata loading for different models
    models_to_test = ['mlp', 'random_forest', 'catboost']
    
    for model_name in models_to_test:
        print(f"\n📋 Testing {model_name}:")
        metadata = load_model_metadata(model_name)
        
        if metadata:
            print(f"  ✅ Metadata loaded")
            print(f"  📊 Data type: {metadata.get('data_type', 'Unknown')}")
            print(f"  🏷️  Model: {metadata.get('model', 'Unknown')}")
            
            if model_name == 'mlp' and 'best_params' in metadata:
                params = metadata['best_params']
                print(f"  🧠 MLP Architecture: {params.get('hidden_layer_sizes', 'Unknown')}")
                print(f"  ⚙️  Alpha: {params.get('alpha', 'Unknown')}")
        else:
            print(f"  ❌ No metadata found")
    
    print(f"\n✅ System information functionality test complete!")

if __name__ == "__main__":
    test_system_info()
