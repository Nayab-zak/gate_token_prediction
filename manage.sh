#!/bin/bash

# manage.sh - Main management script for the ML pipeline
# Usage: ./manage.sh [command] [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show help
show_help() {
    echo "ML Pipeline Management Script"
    echo "========================"
    echo ""
}

# Function to launch business dashboard
launch_business_dashboard() {
    print_colored $BLUE "🏢 Launching Business Analytics Dashboard..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    
    # Find available port
    for port in 8501 8502 8503 8504 8505; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_colored $GREEN "📊 Starting business dashboard on port $port..."
            print_colored $YELLOW "🌐 Dashboard URL: http://localhost:$port"
            print_colored $BLUE "👥 Mode: Business User Interface"
            python -m streamlit run dhashboards/streamlit_business_dashboard.py --server.port $port --server.headless false
            break
        fi
    done
}

# Function to launch developer dashboard
launch_developer_dashboard() {
    print_colored $BLUE "⚙️ Launching Developer Technical Dashboard..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    
    # Find available port
    for port in 8501 8502 8503 8504 8505; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_colored $GREEN "🔧 Starting developer dashboard on port $port..."
            print_colored $YELLOW "🌐 Dashboard URL: http://localhost:$port"
            print_colored $BLUE "👨‍💻 Mode: Developer Technical Interface"
            python -m streamlit run dhashboards/streamlit_developer_dashboard.py --server.port $port --server.headless false
            break
        fi
    done
}

# Function to launch all dashboards
launch_all_dashboards() {
    print_colored $BLUE "🚀 Launching All Streamlit Dashboards..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    python utils/launch_dashboards.py
    
    print_colored $GREEN "✅ All dashboards started"
}

# Function to launch model comparison dashboard
launch_model_comparison_dashboard() {
    print_colored $BLUE "🚀 Launching Model Comparison Dashboard..."
    export PYTHONPATH="${PROJECT_ROOT}"
    # Find available port
    for port in 8501 8502 8503 8504 8505; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_colored $GREEN "📊 Starting model comparison dashboard on port $port..."
            print_colored $YELLOW "🌐 Dashboard URL: http://localhost:$port"
            python -m streamlit run dhashboards/streamlit_model_comparison.py --server.port $port --server.headless false
            break
        fi
    done
}

# hybrid_run() function
hybrid_run() {
    print_colored $BLUE "🔀 Running Hybrid MLP Feature Comparison Pipeline..."
    export PYTHONPATH="${PROJECT_ROOT}"
    print_colored $YELLOW "[1/5] Combining wide and autoencoded features..."
    python scripts/combine_features.py
    print_colored $YELLOW "[2/5] Training MLP on wide features..."
    python agents/train_mlp.py --feature-set wide
    print_colored $YELLOW "[3/5] Training MLP on combined features..."
    python agents/train_mlp.py --feature-set combined
    print_colored $YELLOW "[4/5] Testing MLP on wide features..."
    python agents/train_mlp.py --feature-set wide --test-only
    print_colored $YELLOW "[5/5] Testing MLP on combined features..."
    python agents/train_mlp.py --feature-set combined --test-only
    print_colored $GREEN "✅ Training and testing complete. Comparing results..."
    python scripts/compare_mlp_feature_sets.py
    print_colored $GREEN "📊 Launching model comparison dashboard..."
    launch_model_comparison_dashboard
}

# full_hybrid_pipeline() function
full_hybrid_pipeline() {
    print_colored $BLUE "🚦 Running Full Hybrid MLP Pipeline (no leakage, correct order)..."
    export PYTHONPATH="${PROJECT_ROOT}"
    print_colored $YELLOW "[1/9] Ingesting raw Excel data..."
    python agents/ingestion_agent.py --input data/input/moves.xlsx --output data/preprocessed/moves_raw.csv
    print_colored $YELLOW "[2/9] Preprocessing raw data..."
    python agents/preprocessing_agent.py --input data/preprocessed/moves_raw.csv --output data/preprocessed/moves_clean.csv
    print_colored $YELLOW "[3/9] Feature engineering (wide features)..."
    python agents/feature_agent.py --input data/preprocessed/moves_clean.csv --output data/preprocessed/moves_wide.csv
    print_colored $YELLOW "[4/9] Splitting wide features (chronological)..."
    python agents/data_split_agent.py --input-path data/preprocessed/moves_wide.csv --output-dir data/preprocessed
    print_colored $YELLOW "[5/9] Scaling wide features (fit on train, transform all)..."
    python agents/scaling_agent.py --input data/preprocessed/wide_train.csv --output-dir data/preprocessed --scaler-path models/scaler.pkl --train-ratio 0.7 --val-ratio 0.15
    print_colored $YELLOW "[6/9] Encoding (autoencoder, fit on train+val, transform all)..."
    python agents/encoder_agent.py --train-path data/preprocessed/X_train_scaled.csv --val-path data/preprocessed/X_val_scaled.csv --test-path data/preprocessed/X_test_scaled.csv --model-path models/autoencoder.h5 --output-dir data/encoded_input --encoding-dim 32 --epochs 20
    print_colored $YELLOW "[7/9] Combining wide and encoded features for all splits..."
    python scripts/combine_features.py
    print_colored $YELLOW "[8/9] Training/testing MLP on wide features..."
    python agents/train_mlp.py --feature-set wide
    python agents/train_mlp.py --feature-set wide --test-only
    print_colored $YELLOW "[9/9] Training/testing Hybrid MLP on combined features..."
    python agents/hybrid_mlp.py
    python agents/hybrid_mlp.py --test-only
    print_colored $GREEN "✅ Training and testing complete. Comparing results..."
    python scripts/compare_mlp_feature_sets.py
    print_colored $GREEN "📊 Launching model comparison dashboard..."
    launch_model_comparison_dashboard
}

# Main script logic
case "${1:-help}" in
    "encode-data")
        encode_data
        ;;
    "resume-from")
        if [[ -z "$2" ]]; then
            print_colored $RED "Error: Please specify step number (1-6)"
            show_help
            exit 1
        fi
        resume_from_step "$2"
        ;;
    "train")
        if [[ -z "$2" ]]; then
            print_colored $RED "Error: Please specify model name"
            show_help
            exit 1
        fi
        
        # Check for --hyper-tune flag
        hyper_tune=false
        if [[ "$3" == "--hyper-tune" ]]; then
            hyper_tune=true
        fi
        
        train_model "$2" "$hyper_tune"
        ;;
    "test")
        if [[ -z "$2" ]]; then
            print_colored $RED "Error: Please specify model name"
            show_help
            exit 1
        fi
        
        test_model "$2"
        ;;
    "eda-input")
        analyze_input_data
        ;;
    "eda-output")
        analyze_model_outputs
        ;;
    "eda-notebook")
        launch_eda_notebook
        ;;
    "dashboard-models")
        launch_model_comparison_dashboard
        ;;
    "dashboard-business")
        launch_business_dashboard
        ;;
    "dashboard-developer")
        launch_developer_dashboard
        ;;
    "dashboard-results")
        launch_model_results_dashboard
        ;;
    "dashboards-all")
        launch_all_dashboards
        ;;
    "realtime-watch")
        # Parse watch directory if provided
        watch_dir="data/staging"
        if [[ "$2" == "--watch-dir" && -n "$3" ]]; then
            watch_dir="$3"
        fi
        start_realtime_watch "$watch_dir"
        ;;
    "realtime-dashboard")
        # Parse watch directory if provided
        watch_dir="data/staging"
        if [[ "$2" == "--watch-dir" && -n "$3" ]]; then
            watch_dir="$3"
        fi
        start_realtime_dashboard "$watch_dir"
        ;;
    "realtime-predict")
        if [[ -z "$2" ]]; then
            print_colored $RED "Error: Please specify input file"
            show_help
            exit 1
        fi
        make_single_prediction "$2"
        ;;
    "realtime-cleanup")
        cleanup_realtime_files
        ;;
    "status")
        check_status
        ;;
    "clean")
        clean_files false
        ;;
    "clean-all")
        clean_files true
        ;;
    "test-env")
        test_environment
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    "hybrid_run")
        hybrid_run
        ;;
    "full_hybrid_pipeline")
        full_hybrid_pipeline
        ;;
    *)
        print_colored $RED "Error: Unknown command '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac
