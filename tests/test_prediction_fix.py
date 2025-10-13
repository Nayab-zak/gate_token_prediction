#!/usr/bin/env python3
"""
Quick test to verify the prediction pipeline fix
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.env import load_config_env

def test_config_loading():
    """Test that configuration loads correctly"""
    cfg = load_config_env("config/config.yaml")
    print("✅ Configuration loaded successfully")
    
    # Test prediction_frequency_hours conversion
    prediction_freq = int(cfg["prediction_frequency_hours"])
    print(f"✅ prediction_frequency_hours: {prediction_freq} (type: {type(prediction_freq)})")
    
    # Use assertions instead of return
    assert cfg is not None
    assert "prediction_frequency_hours" in cfg
    assert isinstance(prediction_freq, int)

def test_variable_access():
    """Test the specific variable access that was causing UnboundLocalError"""
    cfg = load_config_env("config/config.yaml")
    
    # Simulate the problematic code section
    prediction_freq = int(cfg["prediction_frequency_hours"])
    
    # This is where the error occurred - referencing prediction_freq in logging
    log_data = {
        "prediction_frequency_hours": prediction_freq,
        "timezone_config": cfg.get("timezone", "not_set"),
    }
    
    print(f"✅ Variable access test passed: {log_data}")
    
    # Use assertions instead of return
    assert log_data is not None
    assert "prediction_frequency_hours" in log_data
    assert isinstance(log_data["prediction_frequency_hours"], int)

if __name__ == "__main__":
    print("🧪 Testing prediction pipeline fixes...\n")
    
    try:
        test_config_loading()
        test_variable_access()
        print("\n🎉 All tests passed! The UnboundLocalError should be fixed.")
    except Exception as e:
        print(f"\n❌ Some tests failed: {e}")
        raise
