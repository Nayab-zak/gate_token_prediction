#!/usr/bin/env python3
"""
Streamlit App Launcher - Start both dashboards
"""

import subprocess
import sys
import time
import yaml
from pathlib import Path

def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'streamlit': {
                'model_results_port': 8501,
                'model_comparison_port': 8502
            }
        }

def start_streamlit_app(script_path, port):
    """Start a Streamlit app"""
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            script_path, 
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(cmd)
        return process
        
    except Exception as e:
        print(f"Error starting {script_path}: {e}")
        return None

def main():
    config = load_config()
    
    print("🚀 Starting Streamlit Model Dashboards...")
    
    # Check if apps exist
    results_app = "streamlit_model_results.py"
    comparison_app = "streamlit_model_comparison.py"
    
    if not Path(results_app).exists():
        print(f"❌ {results_app} not found")
        return
        
    if not Path(comparison_app).exists():
        print(f"❌ {comparison_app} not found")
        return
    
    # Start apps
    results_port = config['streamlit']['model_results_port']
    comparison_port = config['streamlit']['model_comparison_port']
    
    print(f"🌐 Starting Model Results Dashboard on port {results_port}...")
    results_process = start_streamlit_app(results_app, results_port)
    
    time.sleep(2)  # Give first app time to start
    
    print(f"🌐 Starting Model Comparison Dashboard on port {comparison_port}...")
    comparison_process = start_streamlit_app(comparison_app, comparison_port)
    
    if results_process and comparison_process:
        print("\n✅ Both dashboards started successfully!")
        print(f"📊 Model Results: http://localhost:{results_port}")
        print(f"🏆 Model Comparison: http://localhost:{comparison_port}")
        print("\nPress Ctrl+C to stop both apps...")
        
        try:
            # Wait for processes
            results_process.wait()
            comparison_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboards...")
            results_process.terminate()
            comparison_process.terminate()
    else:
        print("❌ Failed to start one or more dashboards")

if __name__ == "__main__":
    main()
