#!/bin/bash

# Ensure project root is in PYTHONPATH for all scripts
export PYTHONPATH="$(pwd):$PYTHONPATH"

set -e

usage() {
  echo "Usage: $0 [prep|features|eda|train|evaluate|plots|predict|pipeline]"
  echo "  prep       - Run data preparation and sanitization"
  echo "  features   - Run feature engineering pipeline"
  echo "  eda        - Run EDA dashboard/report"
  echo "  train      - Train model pipeline"
  echo "  evaluate   - Evaluate model and generate report"
  echo "  plots      - Generate plots/visualizations"
  echo "  predict    - Run prediction on new data"
  echo "  pipeline    - Run the full workflow: prep, features, train, evaluate, plots"
  exit 1
}

if [ $# -eq 0 ]; then
  usage
fi

case "$1" in
  prep)
    echo "[manage.sh] Running data preparation..."
    python scripts/prep_data.py
    ;;
  features)
    echo "[manage.sh] Running feature engineering..."
    python scripts/feature_engineering.py
    ;;
  eda)
    echo "[manage.sh] Running EDA dashboard/report..."
    streamlit run agents/eda_dashboard_agent.py
    ;;
  train)
    echo "[manage.sh] Training model..."
    python scripts/train.py
    ;;
  evaluate)
    echo "[manage.sh] Evaluating model..."
    python scripts/evaluate.py
    ;;
  plots)
    echo "[manage.sh] Generating plots..."
    python scripts/plots.py
    ;;
  predict)
    echo "[manage.sh] Running prediction..."
    python scripts/predict.py
    ;;
  pipeline)
    echo "[manage.sh] Running full pipeline: prep, features, train, evaluate, plots..."
    ./manage.sh prep
    ./manage.sh features
    ./manage.sh train
    ./manage.sh evaluate
    ./manage.sh plots
    ;;
  *)
    usage
    ;;
esac
