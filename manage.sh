#!/bin/bash
# Simple Management Script
# Usage: ./manage.sh [history|realtime]

set -e

COMMAND=$1

# Set project root to script location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Use conda environment sql_ai_agent
export PYTHONPATH="$PROJECT_ROOT"

# Activate conda environment
source /home/wk-12195/miniconda3/bin/activate sql_ai_agent

if [ "$COMMAND" = "history" ]; then
    echo "🚀 Running FULL TRAINING PIPELINE (History)..."
    echo "   Ingestion → Preprocessing → Feature Engineering → Splitting → Training → Evaluation → Push to Vertica"
    python pipeline_orchestrator.py --enable-deployment "${@:2}"

elif [ "$COMMAND" = "realtime" ]; then
    echo "⚡ Running FULL PREDICTION PIPELINE (Real-time)..."
    echo "   Ingestion → Preprocessing → Feature Engineering → Prediction → Push to Vertica"
    python realtime_orchestrator.py "${@:2}"

else
    echo "❌ Unknown command: $COMMAND"
    echo ""
    echo "🎯 USAGE:"
    echo "  ./manage.sh history                     # Full ML training pipeline + deployment"
    echo "  ./manage.sh realtime                    # Full real-time prediction pipeline + deployment"
    echo ""
    echo "💡 Examples:"
    echo "  ./manage.sh history                     # Train model using historical data and deploy"
    echo "  ./manage.sh realtime                    # Generate predictions and deploy to Vertica"
    exit 1
fi
