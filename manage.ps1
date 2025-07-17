# PowerShell script for Windows
# Ensure project root is in PYTHONPATH for all scripts
$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"

$ErrorActionPreference = "Stop"

# Check if virtual environment exists and activate it
$venvPath = "pred_env\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "[manage.ps1] Activating virtual environment..."
    & $venvPath
} else {
    Write-Host "[manage.ps1] Warning: Virtual environment not found at $venvPath"
}

function Show-Usage {
    Write-Host "Usage: .\manage.ps1 [prep|features|eda|train|train-quick|train-opt|train-simple|evaluate|plots|predict|pipeline|mlflow]"
    Write-Host "  prep        - Run data preparation and sanitization"
    Write-Host "  features    - Run feature engineering pipeline"
    Write-Host "  eda         - Run EDA dashboard/report"
    Write-Host "  train       - Train comprehensive model pipeline (GLMs + Trees + Tuning)"
    Write-Host "  train-quick - Train comprehensive pipeline with reduced trials (faster)"
    Write-Host "  train-opt   - Train optimized model pipeline (legacy with MLflow)"
    Write-Host "  train-simple- Train simple model pipeline (no MLflow dependency)"
    Write-Host "  evaluate    - Evaluate model and generate report"
    Write-Host "  plots       - Generate plots/visualizations"
    Write-Host "  predict     - Run prediction on new data"
    Write-Host "  pipeline    - Run the full workflow: prep, features, train, evaluate, plots"
    Write-Host "  mlflow      - Start MLflow tracking server (localhost:5000)"
    exit 1
}

if ($args.Count -eq 0) {
    Show-Usage
}

switch ($args[0]) {
    "prep" {
        Write-Host "[manage.ps1] Running data preparation..."
        python scripts/prep_data.py
    }
    "features" {
        Write-Host "[manage.ps1] Running feature engineering..."
        python scripts/feature_engineering.py
    }
    "eda" {
        Write-Host "[manage.ps1] Running EDA dashboard/report..."
        streamlit run agents/eda_dashboard_agent.py
    }
    "train" {
        $modelName = if ($args.Count -gt 1) { $args[1] } else { "all" }
        if ($modelName -eq "all") {
            Write-Host "[manage.ps1] Training all models..."
            python scripts/comprehensive_train.py --all --mlflow
        } else {
            Write-Host "[manage.ps1] Training model: $modelName..."
            python scripts/comprehensive_train.py --model $modelName --mlflow
        }
    }
    "train-quick" {
        Write-Host "[manage.ps1] Quick training (reduced trials)..."
        python scripts/comprehensive_train.py --quick
    }
    "train-opt" {
        Write-Host "[manage.ps1] Training optimized model..."
        python scripts/optimized_train.py
    }
    "train-simple" {
        Write-Host "[manage.ps1] Training simple model (no MLflow)..."
        python scripts/simple_train.py
    }
    "evaluate" {
        $modelName = if ($args.Count -gt 1) { $args[1] } else { "all" }
        if ($modelName -eq "all") {
            Write-Host "[manage.ps1] Evaluating all models..."
            python scripts/evaluate.py --all --mlflow
        } else {
            Write-Host "[manage.ps1] Evaluating model: $modelName..."
            python scripts/evaluate.py --model $modelName --mlflow
        }
    }
    "evaluate-all" {
        Write-Host "[manage.ps1] Evaluating all models and hybrid..." -ForegroundColor Green
        python scripts/evaluate_predictions.py --hybrid
    }
    "plots" {
        Write-Host "[manage.ps1] Generating plots..."
        python scripts/plots.py
    }
    "predict" {
        $modelName = if ($args.Count -gt 1) { $args[1] } else { "all" }
        if ($modelName -eq "all") {
            Write-Host "[manage.ps1] Running predictions for all models..."
            python scripts/predict.py --all --mlflow
        } else {
            Write-Host "[manage.ps1] Running predictions for model: $modelName..."
            python scripts/predict.py --model $modelName --mlflow
        }
    }
    "pipeline" {
        Write-Host "[manage.ps1] Running full pipeline: prep, features, train, evaluate, plots..."
        .\manage.ps1 prep
        .\manage.ps1 features
        .\manage.ps1 train
        .\manage.ps1 evaluate
        .\manage.ps1 plots
    }
    "mlflow" {
        Write-Host "[manage.ps1] Starting MLflow tracking server..."
        Write-Host "MLflow UI will be available at: http://localhost:5000"
        Write-Host "Press Ctrl+C to stop the server"
        mlflow server --backend-store-uri file:./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    }
    default {
        Show-Usage
    }
}
