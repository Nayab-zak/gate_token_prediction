#!/usr/bin/env bash
set -e

ENV_NAME="sql_ai_agent"
REQUIREMENTS="requirements.txt"

# Check if conda is available
if command -v conda &> /dev/null; then
    # Check if env exists
    if conda info --envs | grep -q "^$ENV_NAME[[:space:]]"; then
        echo "Conda environment $ENV_NAME already exists."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
    else
        echo "Creating conda environment $ENV_NAME..."
        conda create -y -n $ENV_NAME python=3.10
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        pip install -r $REQUIREMENTS
    fi
else
    # Fallback to python venv
    if [ -d ".venv" ]; then
        echo "Python venv already exists. Activating..."
        source .venv/bin/activate
    else
        echo "Creating python venv..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r $REQUIREMENTS
    fi
fi
