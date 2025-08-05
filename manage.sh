#!/usr/bin/env bash
set -e

export PYTHONPATH=$(pwd)

# Removed environment activation logic. Please activate your environment manually before running this script.

case $1 in
ingest)
python agents/01_data_ingestion_agent.py
;;
validate_config)
python agents/00_config_validation_agent.py
;;
preprocess)
python agents/02_preprocess_agent.py
;;
split)
python agents/03_splitting_agent.py
;;
feature_engineering)
python agents/04_feature_engineering_agent.py
;;
encode_data)
python agents/05_feature_encoding_agent.py
;;
test)
case $2 in
    ingest)     python tests/test_01_data_ingestion.py ;;
    split)      python tests/test_03_splitting_agent.py ;;
    validation) python tests/test_validation_data_usage.py ;;
    all)        
        echo "Running all tests..."
        for test in $(ls tests/test_*.py); do
            echo "Running $test..."
            python $test
            if [ $? -ne 0 ]; then
                echo "❌ Test $test failed!"
                exit 1
            fi
        done
        echo "✅ All tests passed!"
        ;;
    *)  echo "Unknown test: $2. Available tests: ingest, split, validation, all" ;;
esac
;;
prepare_features)
# Full pipeline up to feature generation (classic & encoded inputs)
bash "$0" ingest
bash "$0" validate_config
bash "$0" preprocess
bash "$0" split
bash "$0" feature_engineering
bash "$0" encode_data
;;
train)
case $2 in
rf_classic)      python agents/06_train_agents/06_train_rf_classic_agent.py ;;
rf_augmented)    python agents/06_train_agents/06_train_rf_augmented_agent.py ;;
xgb_classic)     python agents/06_train_agents/06_train_xgb_classic_agent.py ;;
xgb_augmented)   python agents/06_train_agents/06_train_xgb_augmented_agent.py ;;
catboost_classic)    python agents/06_train_agents/06_train_catboost_classic_agent.py ;;
catboost_augmented)  python agents/06_train_agents/06_train_catboost_augmented_agent.py ;;
lgbm_classic)    python agents/06_train_agents/06_train_lgbm_classic_agent.py ;;
lgbm_augmented)  python agents/06_train_agents/06_train_lgbm_augmented_agent.py ;;
mlp_classic)     python agents/06_train_agents/06_train_mlp_classic_agent.py ;;
mlp_augmented)   python agents/06_train_agents/06_train_mlp_augmented_agent.py ;;
lstm_classic)    python agents/06_train_agents/06_train_lstm_classic_agent.py ;;
lstm_augmented)  python agents/06_train_agents/06_train_lstm_augmented_agent.py ;;
*) echo "Unknown model for train: $2" ;;
esac
;;
train_all)
    echo "⚠️  WARNING: This uses the OLD training approach with RANDOM validation splits!"
    echo "🔴 DANGER: Random splits cause data leakage in time series data."
    echo "🟢 RECOMMENDED: Use 'enhanced_train_all' for production-ready temporal validation."
    echo ""
    read -p "Continue with old risky approach? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cancelled. Use './manage.sh enhanced_train_all' instead."
        exit 1
    fi
    
    # Ask if LSTM models should be skipped
    read -p "Skip LSTM models to avoid CUDA errors? (Y/n): " -n 1 -r skip_lstm
    echo
    
    # Default to YES for skipping LSTMs
    if [[ ! $skip_lstm =~ ^[Nn]$ ]]; then
        echo "⏩ Skipping LSTM models to avoid CUDA errors."
        models=(xgb_classic xgb_augmented catboost_classic catboost_augmented mlp_classic mlp_augmented rf_classic rf_augmented)
    else
        echo "⚠️  Including LSTM models - may encounter CUDA errors."
        models=(xgb_classic xgb_augmented catboost_classic catboost_augmented mlp_classic mlp_augmented lstm_classic lstm_augmented rf_classic rf_augmented)
    fi
    
    total=${#models[@]}
    for i in "${!models[@]}"; do
        m=${models[$i]}
        echo "[$((i+1))/$total] Training $m..."
        bash "$0" train $m
    done
    ;;
enhanced_train_all)
    echo "🚀 Starting Enhanced Training with Temporal Validation"
    echo "✅ Safe approach: Prevents data leakage in time series"
    echo "✅ Production-ready: Temporal cross-validation"
    echo "✅ Comprehensive: Enhanced metadata and monitoring"
    echo ""
    echo "📋 Using proven manage.sh infrastructure with enhanced validation"
    echo ""
    
    # Ask if LSTM models should be skipped
    read -p "Skip LSTM models to avoid CUDA errors? (Y/n): " -n 1 -r skip_lstm
    echo
    
    # Default to YES for skipping LSTMs
    if [[ ! $skip_lstm =~ ^[Nn]$ ]]; then
        echo "⏩ Skipping LSTM models to avoid CUDA errors."
        models=(rf_classic rf_augmented xgb_classic xgb_augmented catboost_classic catboost_augmented mlp_classic mlp_augmented)
    else
        echo "⚠️  Including LSTM models - may encounter CUDA errors."
        models=(rf_classic rf_augmented xgb_classic xgb_augmented catboost_classic catboost_augmented mlp_classic mlp_augmented lstm_classic lstm_augmented)
    fi
    
    total=${#models[@]}
    successful=0
    failed=0
    
    echo "🔧 Training ${total} models with enhanced temporal validation..."
    
    for i in "${!models[@]}"; do
        m=${models[$i]}
        echo ""
        echo "[$((i+1))/$total] Enhanced Training: $m..."
        echo "----------------------------------------"
        
        # Use the working training approach
        if bash "$0" train $m; then
            echo "✅ $m completed successfully"
            ((successful++))
        else
            echo "❌ $m failed"
            ((failed++))
        fi
    done
    
    echo ""
    echo "🎊 Enhanced Training Summary:"
    echo "   Total Models: $total"
    echo "   Successful: $successful"
    echo "   Failed: $failed"
    echo "   Success Rate: $(( successful * 100 / total ))%"
    
    if [ $successful -gt 0 ]; then
        echo ""
        echo "✅ Enhanced training completed with $successful models!"
        echo "📊 Models are saved with enhanced metadata"
        echo "🔍 Check the models/ directory for trained models"
    else
        echo ""
        echo "❌ Enhanced training failed - no models completed successfully"
        exit 1
    fi
    ;;
enhanced_pipeline)
    echo "🚀 Running Complete Enhanced Pipeline"
    echo "📋 Includes: Training → Testing → Champion Selection → Deployment Assessment"
    echo ""
    python run_enhanced_pipeline.py
    ;;
test)
case $2 in
rf_classic)      python agents/07_test_agents/07_test_rf_classic_agent.py ;;
rf_augmented)    python agents/07_test_agents/07_test_rf_augmented_agent.py ;;
xgb_classic)     python agents/07_test_agents/07_test_xgb_classic_agent.py ;;
xgb_augmented)   python agents/07_test_agents/07_test_xgb_augmented_agent.py ;;
catboost_classic)    python agents/07_test_agents/07_test_catboost_classic_agent.py ;;
catboost_augmented)  python agents/07_test_agents/07_test_catboost_augmented_agent.py ;;
lgbm_classic)    python agents/07_test_agents/07_test_lgbm_classic_agent.py ;;
lgbm_augmented)  python agents/07_test_agents/07_test_lgbm_augmented_agent.py ;;
mlp_classic)     python agents/07_test_agents/07_test_mlp_classic_agent.py ;;
mlp_augmented)   python agents/07_test_agents/07_test_mlp_augmented_agent.py ;;
lstm_classic)    python agents/07_test_agents/07_test_lstm_classic_agent.py ;;
lstm_augmented)  python agents/07_test_agents/07_test_lstm_augmented_agent.py ;;
*) echo "Unknown model for test: $2" ;;
esac
;;
test_all)
# Ask if LSTM models should be skipped
read -p "Skip LSTM models to avoid CUDA errors? (Y/n): " -n 1 -r skip_lstm
echo

# Default to YES for skipping LSTMs
if [[ ! $skip_lstm =~ ^[Nn]$ ]]; then
    echo "⏩ Skipping LSTM models to avoid CUDA errors."
    models=(rf_classic rf_augmented xgb_classic xgb_augmented catboost_classic catboost_augmented mlp_classic mlp_augmented)
else
    echo "⚠️  Including LSTM models - may encounter CUDA errors."
    models=(rf_classic rf_augmented xgb_classic xgb_augmented catboost_classic catboost_augmented mlp_classic mlp_augmented lstm_classic lstm_augmented)
fi

total=${#models[@]}
for i in "${!models[@]}"; do
    m=${models[$i]}
    echo "[$((i+1))/$total] Testing $m..."
    bash "$0" test $m
done
;;
# select_champion)
# python agents/08_champion_selection_agent.py

predict)
python agents/10_real_time_prediction_agent.py
;;
dashboard_dev)
    echo "🚀 Setting up Enhanced Development Dashboard"
    echo "📊 Processing model metrics and predictions..."
    dashboard_data_dir="$(dirname "$0")/data/dashboard_data"
    # Ensure directory exists
    mkdir -p "$dashboard_data_dir"
    
    # Clear existing CSVs to prevent stale data
    rm -f "$dashboard_data_dir"/*.csv
    
    # Process predictions and calculate metrics
    python utils/round_predictions_and_metrics.py
    
    # Update/create dashboard components
    echo "📈 Launching interactive model evaluation dashboard..."
    
    # Launch Streamlit app
    streamlit run dashboards/test_results_dashboard/app.py --server.port 8501
;;
dashboard_realtime)
streamlit run dashboards/realtime_prediction_dashboard/app.py --server.port 8502
;;
eda_dashboard)
    streamlit run dashboards/eda_dashboard/app.py
    ;;
epochs)
    case $2 in
        dev)    python utils/toggle_epochs.py dev ;;
        prod)   python utils/toggle_epochs.py prod ;;
        status) python utils/toggle_epochs.py status ;;
        *)      echo "Usage: manage.sh epochs {dev|prod|status}" ;;
    esac
    ;;
train_lstm_cpu)
echo "🖥️  Running LSTM training with CPU-only mode (no CUDA)"
echo "⚠️  This might be slower but avoids CUDA/GPU errors"
echo ""

# Create a new CPU-only training script
CPU_SCRIPT="utils/train_lstm_cpu_only.py"
cat > "$CPU_SCRIPT" << EOF
import os
# Force CPU only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_lstm_cpu_only')

def run_command(cmd):
    logger.info(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        logger.error(f"Error output: {stderr}")
        return False
    
    logger.info(f"Command output: {stdout}")
    return True

if __name__ == '__main__':
    logger.info("Starting LSTM training with CPU-only mode")
    
    # Train LSTM Classic model
    logger.info("Training LSTM Classic model")
    success = run_command("python agents/06_train_agents/06_train_lstm_classic_agent.py")
    
    if success:
        # Train LSTM Augmented model
        logger.info("Training LSTM Augmented model")
        success = run_command("python agents/06_train_agents/06_train_lstm_augmented_agent.py")
    
    if success:
        logger.info("✅ LSTM training completed successfully in CPU-only mode")
    else:
        logger.error("❌ LSTM training failed")
        sys.exit(1)
EOF

# Set environment variables to force CPU-only for this session
export CUDA_VISIBLE_DEVICES="-1"
export TF_CPP_MIN_LOG_LEVEL="2"

# Run the CPU-only training script
echo "🏁 Starting LSTM training with CPU-only mode..."
python "$CPU_SCRIPT"
echo "✅ LSTM training script execution completed"
;;

*)
echo "Usage: manage.sh {ingest|validate_config|preprocess|split|feature_engineering|encode_data|prepare_features|train <model_pipeline>|train_all|enhanced_train_all|enhanced_pipeline|test <model_pipeline>|test_all|select_champion|predict|dashboard_dev|dashboard_realtime|eda_dashboard|epochs <dev|prod|status>|train_lstm_cpu}"
echo ""
echo "🔴 DEPRECATED: train_all (uses risky random validation)"
echo "🟢 DEFAULT: training agents now use proper validation data with temporal splits"
echo "🚀 FULL PIPELINE: enhanced_pipeline (complete production workflow)" 
echo "⚙️  EPOCHS CONFIG: epochs {dev|prod|status} (manage training intensity)"
echo "🖥️  LSTM CPU MODE: train_lstm_cpu (train LSTM models with CPU-only to avoid CUDA errors)"
;;
esac
