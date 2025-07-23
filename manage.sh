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
            python -m streamlit run streamlit_business_dashboard.py --server.port $port --server.headless false
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
            python -m streamlit run streamlit_developer_dashboard.py --server.port $port --server.headless false
            break
        fi
    done
}

# Function to launch all dashboards
launch_all_dashboards() {
    print_colored $BLUE "🚀 Launching All Dashboards..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    
    # Start business dashboard
    print_colored $GREEN "📊 Starting business dashboard on port 8501..."
    python -m streamlit run streamlit_business_dashboard.py --server.port 8501 --server.headless true &
    
    # Start developer dashboard
    print_colored $GREEN "🔧 Starting developer dashboard on port 8502..."
    python -m streamlit run streamlit_developer_dashboard.py --server.port 8502 --server.headless true &
    
    # Start original model comparison
    print_colored $GREEN "📈 Starting model comparison dashboard on port 8503..."
    python -m streamlit run streamlit_model_comparison.py --server.port 8503 --server.headless true &
    
    sleep 3
    print_colored $YELLOW "🌐 Dashboard URLs:"
    print_colored $YELLOW "   Business Dashboard: http://localhost:8501"
    print_colored $YELLOW "   Developer Dashboard: http://localhost:8502"
    print_colored $YELLOW "   Model Comparison: http://localhost:8503"
    
    print_colored $GREEN "✅ All dashboards launched successfully!"
}

# Function to show help
show_help() {
    echo "ML Pipeline Management Script"
    echo "========================"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Data Pipeline Commands:"
    echo "  encode-data              Run full pipeline to encode data"
    echo "  resume-from [step]       Resume pipeline from specific step (1-6)"
    echo "  status                   Show pipeline status and file information"
    echo "  clean                    Clean generated files (keeps input data)"
    echo "  clean-all                Clean all generated files including models"
    echo "  test-env                 Test environment and dependencies"
    echo "  help                     Show this help message"
    echo ""
    echo "Model Training Commands:"
    echo "  train [model] [--hyper-tune]      Train specific model or all models"
    echo "  test [model]                      Test specific model or all models"
    echo ""
    echo "EDA and Analysis Commands:"
    echo "  eda-input                         Analyze input data structure and statistics"
    echo "  eda-output                        Interactive model results and predictions analysis"
    echo "  eda-notebook                      Launch Jupyter notebook for detailed analysis"
    echo "  dashboard-business                Launch business-friendly analytics dashboard"
    echo "  dashboard-developer               Launch technical developer dashboard"
    echo "  dashboards-all                    Launch all Streamlit dashboards"
    echo ""
    echo "Realtime Prediction Commands:"
    echo "  realtime-watch [--watch-dir DIR]  Start realtime prediction monitoring"
    echo "  realtime-predict FILE             Make prediction on a single file"
    echo "  realtime-dashboard                Launch Streamlit dashboard for monitoring"
    echo "  realtime-cleanup                  Clean up temporary realtime files"
    echo ""
    echo "Available Models:"
    echo "  all                      Train/test all available models"
    echo "  elasticnet              ElasticNet regression model"
    echo "  catboost                CatBoost gradient boosting model"
    echo "  extra_trees             Extra Trees ensemble model"
    echo "  random_forest           Random Forest ensemble model"
    echo "  lightgbm                LightGBM gradient boosting model"
    echo "  xgboost                 XGBoost gradient boosting model"
    echo "  mlp                     Multi-Layer Perceptron neural network"
    echo ""
    echo "Data Pipeline Steps:"
    echo "  1. ingestion             Excel to raw CSV"
    echo "  2. preprocessing         Data cleaning and datetime parsing"
    echo "  3. aggregation           Grouping and wide format conversion"
    echo "  4. feature               Feature engineering (time, lags, rolling)"
    echo "  5. scaling               Data scaling and train/val/test split"
    echo "  6. encoder               Autoencoder training and embedding creation"
    echo ""
    echo "Examples:"
    echo "  $0 encode-data                    # Run complete data pipeline"
    echo "  $0 train all --hyper-tune        # Train all models with hyperparameter tuning"
    echo "  $0 train catboost --hyper-tune   # Train CatBoost with hyperparameter tuning"
    echo "  $0 train random_forest           # Train Random Forest without hyperparameter tuning"
    echo "  $0 test all                      # Test all trained models"
    echo "  $0 test elasticnet               # Test ElasticNet model only"
    echo "  $0 eda-input                     # Analyze input data structure and statistics"
    echo "  $0 eda-output                    # Interactive analysis of model predictions"
    echo "  $0 eda-notebook                  # Launch Jupyter notebook for detailed analysis"
    echo "  $0 dashboard-business            # Launch business-friendly analytics dashboard"
    echo "  $0 dashboard-developer           # Launch technical developer dashboard"
    echo "  $0 dashboards-all                # Launch all Streamlit dashboards"
    echo "  $0 resume-from 3                 # Resume pipeline from aggregation step"
    echo "  $0 realtime-watch                # Start realtime prediction monitoring"
    echo "  $0 realtime-predict data.csv     # Make prediction on single file"
    echo "  $0 realtime-dashboard            # Launch prediction dashboard"
    echo "  $0 realtime-cleanup              # Clean up temporary realtime files"
    echo "  $0 status                        # Check current status"
    echo "  $0 clean                         # Clean intermediate files"
}

# Function to test environment
test_environment() {
    print_colored $BLUE "🔍 Testing environment and dependencies..."
    
    # Check if we're in conda environment
    if [[ "$CONDA_DEFAULT_ENV" == "sql_ai_agent" ]]; then
        print_colored $GREEN "✓ conda environment: $CONDA_DEFAULT_ENV"
    else
        print_colored $YELLOW "⚠ Warning: Not in sql_ai_agent conda environment"
    fi
    
    # Check Python packages
    print_colored $BLUE "Checking Python packages..."
    python -c "
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import joblib
print('✓ pandas:', pd.__version__)
print('✓ numpy:', np.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ tensorflow:', tf.__version__)
print('✓ joblib: available')
"
    
    # Check input file
    if [[ -f "data/input/moves.xlsx" ]]; then
        print_colored $GREEN "✓ Input file exists: data/input/moves.xlsx"
        python -c "
import pandas as pd
df = pd.read_excel('data/input/moves.xlsx')
print(f'✓ Input file readable: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'✓ Columns: {list(df.columns)}')
"
    else
        print_colored $RED "✗ Input file missing: data/input/moves.xlsx"
        exit 1
    fi
    
    print_colored $GREEN "🎉 Environment test completed successfully!"
}

# Function to check pipeline status
check_status() {
    print_colored $BLUE "📊 Pipeline Status Report"
    echo "========================"
    
    # Check input files
    echo ""
    print_colored $BLUE "Input Files:"
    if [[ -f "data/input/moves.xlsx" ]]; then
        size=$(du -h "data/input/moves.xlsx" | cut -f1)
        print_colored $GREEN "✓ data/input/moves.xlsx ($size)"
    else
        print_colored $RED "✗ data/input/moves.xlsx (missing)"
    fi
    
    # Check intermediate files
    echo ""
    print_colored $BLUE "Intermediate Files:"
    intermediate_files=(
        "data/preprocessed/moves_raw.csv"
        "data/preprocessed/moves_clean.csv"
        "data/preprocessed/moves_wide.csv"
        "data/preprocessed/moves_features.csv"
        "data/preprocessed/X_train_scaled.csv"
        "data/preprocessed/X_val_scaled.csv"
        "data/preprocessed/X_test_scaled.csv"
    )
    
    for file in "${intermediate_files[@]}"; do
        if [[ -f "$file" ]]; then
            size=$(du -h "$file" | cut -f1)
            print_colored $GREEN "✓ $file ($size)"
        else
            print_colored $YELLOW "○ $file (not created)"
        fi
    done
    
    # Check model files
    echo ""
    print_colored $BLUE "Model Files:"
    model_files=(
        "models/scaler.pkl"
        "models/autoencoder.h5"
        "models/training_history.csv"
    )
    
    for file in "${model_files[@]}"; do
        if [[ -f "$file" ]]; then
            size=$(du -h "$file" | cut -f1)
            print_colored $GREEN "✓ $file ($size)"
        else
            print_colored $YELLOW "○ $file (not created)"
        fi
    done
    
    # Check output files
    echo ""
    print_colored $BLUE "Output Files (Encoded Data):"
    output_files=(
        "data/encoded_input/Z_train.csv"
        "data/encoded_input/Z_val.csv"
        "data/encoded_input/Z_test.csv"
    )
    
    all_outputs_exist=true
    for file in "${output_files[@]}"; do
        if [[ -f "$file" ]]; then
            size=$(du -h "$file" | cut -f1)
            rows=$(tail -n +2 "$file" | wc -l)
            print_colored $GREEN "✓ $file ($size, $rows rows)"
        else
            print_colored $YELLOW "○ $file (not created)"
            all_outputs_exist=false
        fi
    done
    
    # Check logs
    echo ""
    print_colored $BLUE "Log Files:"
    if [[ -d "logs" ]]; then
        log_count=$(find logs -name "*.log" | wc -l)
        print_colored $GREEN "✓ logs directory ($log_count log files)"
    else
        print_colored $YELLOW "○ logs directory (not created)"
    fi
    
    # Summary
    echo ""
    if [[ "$all_outputs_exist" == true ]]; then
        print_colored $GREEN "🎉 Pipeline appears to be complete! Encoded data is ready."
    else
        print_colored $YELLOW "⏳ Pipeline is incomplete. Run 'encode-data' or 'resume-from [step]'."
    fi
}

# Function to clean generated files
clean_files() {
    local clean_all=${1:-false}
    
    print_colored $YELLOW "🧹 Cleaning generated files..."
    
    # Clean intermediate files
    if [[ -d "data/preprocessed" ]]; then
        find data/preprocessed -name "*.csv" -delete 2>/dev/null || true
        print_colored $BLUE "✓ Cleaned preprocessed data files"
    fi
    
    # Clean output files
    if [[ -d "data/encoded_input" ]]; then
        find data/encoded_input -name "*.csv" -delete 2>/dev/null || true
        print_colored $BLUE "✓ Cleaned encoded output files"
    fi
    
    if [[ -d "data/encoded_output" ]]; then
        find data/encoded_output -name "*.csv" -delete 2>/dev/null || true
        print_colored $BLUE "✓ Cleaned encoded output directory"
    fi
    
    # Clean logs
    if [[ -d "logs" ]]; then
        find logs -name "*.log" -delete 2>/dev/null || true
        print_colored $BLUE "✓ Cleaned log files"
    fi
    
    # Clean models if requested
    if [[ "$clean_all" == true ]]; then
        if [[ -d "models" ]]; then
            find models -name "*" -type f -delete 2>/dev/null || true
            print_colored $BLUE "✓ Cleaned model files"
        fi
    fi
    
    print_colored $GREEN "🎉 Cleaning completed!"
}

# Function to run the pipeline
encode_data() {
    print_colored $GREEN "🚀 Starting ML Pipeline: Raw Data → Encoded Embeddings"
    echo "======================================================"
    
    # Test environment first
    test_environment
    
    echo ""
    print_colored $BLUE "📋 Pipeline Steps:"
    echo "1. 📥 Ingestion: Excel → Raw CSV"
    echo "2. 🧹 Preprocessing: Clean & Parse Dates"
    echo "3. 📊 Aggregation: Group & Pivot to Wide Format"
    echo "4. ⚙️ Feature Engineering: Time Features, Lags, Rolling Stats"
    echo "5. 📏 Scaling: Split & Scale Data"
    echo "6. 🧠 Encoder: Train Autoencoder & Create Embeddings"
    
    echo ""
    print_colored $YELLOW "⏰ This may take several minutes to complete..."
    
    # Run the orchestrator
    python agents/orchestrator_agent.py --start-from 1 --log-dir logs
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo ""
        print_colored $GREEN "🎉 SUCCESS! Pipeline completed successfully!"
        echo ""
        print_colored $BLUE "📁 Encoded data is now available in:"
        echo "  - data/encoded_input/Z_train.csv"
        echo "  - data/encoded_input/Z_val.csv"
        echo "  - data/encoded_input/Z_test.csv"
        echo ""
        print_colored $BLUE "🎯 You can now use the encoded data for your predictive models!"
        
        # Show final status
        check_status
    else
        print_colored $RED "💥 Pipeline failed! Check logs for details:"
        echo "  - logs/orchestrator_agent.log"
        echo "  - logs/[agent_name].log"
        exit $exit_code
    fi
}

# Function to resume pipeline from specific step
resume_from_step() {
    local step=$1
    
    if [[ ! "$step" =~ ^[1-6]$ ]]; then
        print_colored $RED "Error: Step must be between 1-6"
        exit 1
    fi
    
    step_names=("" "ingestion" "preprocessing" "aggregation" "feature" "scaling" "encoder")
    
    print_colored $GREEN "🔄 Resuming pipeline from step $step (${step_names[$step]})"
    
    python agents/orchestrator_agent.py --start-from $step --log-dir logs
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        print_colored $GREEN "🎉 Pipeline resumed and completed successfully!"
        check_status
    else
        print_colored $RED "💥 Pipeline failed! Check logs for details."
        exit $exit_code
    fi
}

# Function to validate model name
validate_model_name() {
    local model=$1
    local valid_models=("all" "elasticnet" "catboost" "extra_trees" "random_forest" "lightgbm" "xgboost" "mlp")
    
    for valid_model in "${valid_models[@]}"; do
        if [[ "$model" == "$valid_model" ]]; then
            return 0
        fi
    done
    
    print_colored $RED "Error: Invalid model name '$model'"
    echo "Valid models: ${valid_models[*]}"
    exit 1
}

# Function to train a specific model
train_model() {
    local model=$1
    local hyper_tune=${2:-false}
    
    validate_model_name "$model"
    
    # Set up environment
    export PYTHONPATH="${PROJECT_ROOT}"
    
    if [[ "$hyper_tune" == true ]]; then
        print_colored $GREEN "🚀 Training $model with hyperparameter tuning..."
    else
        print_colored $GREEN "🚀 Training $model without hyperparameter tuning..."
    fi
    
    if [[ "$model" == "all" ]]; then
        local all_models=("elasticnet" "catboost" "extra_trees" "random_forest" "lightgbm" "xgboost" "mlp")
        
        for single_model in "${all_models[@]}"; do
            echo ""
            print_colored $BLUE "📊 Training $single_model..."
            
            if [[ "$hyper_tune" == true ]]; then
                python "agents/train_${single_model}.py" --hyper-tune
            else
                python "agents/train_${single_model}.py"
            fi
            
            local exit_code=$?
            if [[ $exit_code -eq 0 ]]; then
                print_colored $GREEN "✅ $single_model training completed successfully!"
            else
                print_colored $RED "❌ $single_model training failed!"
                exit $exit_code
            fi
        done
        
        print_colored $GREEN "🎉 All models trained successfully!"
    else
        if [[ "$hyper_tune" == true ]]; then
            python "agents/train_${model}.py" --hyper-tune
        else
            python "agents/train_${model}.py"
        fi
        
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            print_colored $GREEN "🎉 $model training completed successfully!"
        else
            print_colored $RED "💥 $model training failed!"
            exit $exit_code
        fi
    fi
}

# Function to test a specific model
test_model() {
    local model=$1
    
    validate_model_name "$model"
    
    # Set up environment
    export PYTHONPATH="${PROJECT_ROOT}"
    
    print_colored $GREEN "🧪 Testing $model..."
    
    if [[ "$model" == "all" ]]; then
        local all_models=("elasticnet" "catboost" "extra_trees" "random_forest" "lightgbm" "xgboost" "mlp")
        
        for single_model in "${all_models[@]}"; do
            echo ""
            print_colored $BLUE "📊 Testing $single_model..."
            
            python "agents/train_${single_model}.py" --test-only
            
            local exit_code=$?
            if [[ $exit_code -eq 0 ]]; then
                print_colored $GREEN "✅ $single_model testing completed successfully!"
            else
                print_colored $RED "❌ $single_model testing failed!"
                exit $exit_code
            fi
        done
        
        print_colored $GREEN "🎉 All models tested successfully!"
    else
        python "agents/train_${model}.py" --test-only
        
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            print_colored $GREEN "🎉 $model testing completed successfully!"
        else
            print_colored $RED "💥 $model testing failed!"
            exit $exit_code
        fi
    fi
}

# Function to start realtime prediction monitoring
start_realtime_watch() {
    local watch_dir="${1:-data/staging}"
    
    print_colored $BLUE "🚀 Starting realtime prediction monitoring..."
    print_colored $YELLOW "📁 Staging directory: $watch_dir"
    
    # Check if champion model exists
    if [[ ! -f "models/champion.txt" ]]; then
        print_colored $RED "❌ No champion model selected. Please train and set a champion model first."
        exit 1
    fi
    
    # Start realtime agent
    export PYTHONPATH="${PROJECT_ROOT}"
    python agents/realtime_predict_agent.py --staging-dir "$watch_dir"
}

# Function to start realtime prediction with dashboard
start_realtime_dashboard() {
    local watch_dir="${1:-data/staging}"
    
    print_colored $BLUE "🚀 Starting realtime prediction with dashboard..."
    print_colored $YELLOW "📁 Staging directory: $watch_dir"
    print_colored $YELLOW "🌐 Dashboard will be available at: http://localhost:8503"
    
    # Check if champion model exists
    if [[ ! -f "models/champion.txt" ]]; then
        print_colored $RED "❌ No champion model selected. Please train and set a champion model first."
        exit 1
    fi
    
    # Start realtime agent with dashboard
    export PYTHONPATH="${PROJECT_ROOT}"
    python agents/realtime_predict_agent.py --staging-dir "$watch_dir" --dashboard
}

# Function to make prediction on a single file
make_single_prediction() {
    local input_file="$1"
    
    if [[ -z "$input_file" ]]; then
        print_colored $RED "❌ Please specify input file"
        exit 1
    fi
    
    if [[ ! -f "$input_file" ]]; then
        print_colored $RED "❌ Input file not found: $input_file"
        exit 1
    fi
    
    print_colored $BLUE "🔮 Making prediction on: $input_file"
    
    # Check if champion model exists
    if [[ ! -f "models/champion.txt" ]]; then
        print_colored $RED "❌ No champion model selected. Please train and set a champion model first."
        exit 1
    fi
    
    # Create staging directory if it doesn't exist
    mkdir -p data/staging
    
    # Copy file to staging (this will trigger processing)
    local staging_file="data/staging/$(basename "$input_file")"
    cp "$input_file" "$staging_file"
    
    print_colored $GREEN "✅ File copied to staging directory. Processing will begin automatically if realtime agent is running."
    print_colored $YELLOW "💡 To start realtime monitoring: $0 realtime-watch"
    print_colored $YELLOW "💡 To process immediately: $0 realtime-watch (in another terminal)"
}

# Function to cleanup realtime prediction files
cleanup_realtime_files() {
    print_colored $BLUE "🧹 Cleaning up realtime prediction files..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    python agents/realtime_predict_agent.py --cleanup
    
    print_colored $GREEN "✅ Cleanup completed"
}

# Function to analyze input data structure
analyze_input_data() {
    print_colored $BLUE "📊 Analyzing Input Data Structure and Statistics..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    python examine_data.py
    
    print_colored $GREEN "✅ Input data analysis completed"
}

# Function to run interactive model output analysis
analyze_model_outputs() {
    print_colored $BLUE "🔍 Starting Interactive Model Output Analysis..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    python model_explorer.py
    
    print_colored $GREEN "✅ Model output analysis completed"
}

# Function to launch Jupyter notebook for detailed EDA
launch_eda_notebook() {
    print_colored $BLUE "📓 Launching Jupyter Notebook for Detailed Analysis..."
    
    # Check if jupyter is available
    if command -v jupyter >/dev/null 2>&1; then
        print_colored $GREEN "🚀 Opening model_analysis_eda.ipynb..."
        jupyter notebook model_analysis_eda.ipynb
    else
        print_colored $YELLOW "⚠ Jupyter not found. Opening notebook file for viewing..."
        print_colored $BLUE "📂 Notebook location: model_analysis_eda.ipynb"
        print_colored $BLUE "💡 Install Jupyter: pip install jupyter"
    fi
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
            python -m streamlit run streamlit_model_comparison.py --server.port $port --server.headless false
            break
        fi
    done
}

# Function to launch model results dashboard
launch_model_results_dashboard() {
    print_colored $BLUE "🚀 Launching Model Results Dashboard..."
    
    export PYTHONPATH="${PROJECT_ROOT}"
    
    # Find available port
    for port in 8503 8504 8505 8506 8507; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_colored $GREEN "📈 Starting model results dashboard on port $port..."
            print_colored $YELLOW "🌐 Dashboard URL: http://localhost:$port"
            python -m streamlit run streamlit_model_results.py --server.port $port --server.headless false
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
    *)
        print_colored $RED "Error: Unknown command '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac
