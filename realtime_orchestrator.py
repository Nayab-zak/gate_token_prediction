#!/usr/bin/env python3
"""
Real-time Pipeline Orchestrator for Gate Token Prediction
========================================================

This orchestrator manages the real-time prediction pipeline:
1. Ingestion: Get latest data from Vertica
2. Preprocessing: Clean and standardize the data
3. Feature Engineering: Build features for prediction
4. Prediction: Use trained model to generate predictions
5. Push: Append predictions to Vertica table

Key differences from training pipeline:
- No data splitting (uses all latest data)
- No model training (uses existing trained model)
- Focuses on speed and reliability for production use
- Automatic deployment to prediction table
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pandas as pd

from config import settings
from utils.logging import get_logger

# Import individual agents
from agents import (
    ingestion_agent,
    preprocessing_agent,
    feature_engineering_agent,
    vertica_push_agent
)

# Import CatBoost for prediction
from catboost import CatBoostRegressor

logger = get_logger(__name__)

@dataclass
class RealtimePipelineStage:
    name: str
    description: str
    required_inputs: List[str]
    expected_outputs: List[str]
    agent_function: callable
    enabled: bool = True
    
@dataclass 
class RealtimePipelineResult:
    stage: str
    success: bool
    execution_time: float
    input_files: List[str]
    output_files: List[str]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class RealtimePipelineOrchestrator:
    """
    Real-time pipeline orchestrator for production predictions.
    Optimized for speed and reliability in live environments.
    """
    
    def __init__(self):
        self.results: List[RealtimePipelineResult] = []
        self.start_time = datetime.now()
        
        # Define real-time pipeline stages
        self.stages = [
            RealtimePipelineStage(
                name="ingestion",
                description="Extract latest data from Vertica database",
                required_inputs=[],
                expected_outputs=["data/input_raw/realtime/*.csv"],
                agent_function=self._run_ingestion
            ),
            RealtimePipelineStage(
                name="preprocessing", 
                description="Clean and standardize real-time data",
                required_inputs=["data/input_raw/realtime/*.csv"],
                expected_outputs=["data/preprocessed/realtime/*.csv"],
                agent_function=self._run_preprocessing
            ),
            RealtimePipelineStage(
                name="feature_engineering",
                description="Build features for prediction (no splitting)",
                required_inputs=["data/preprocessed/realtime/*.csv"],
                expected_outputs=["data/features/realtime/*.parquet"],
                agent_function=self._run_feature_engineering
            ),
            RealtimePipelineStage(
                name="prediction",
                description="Generate predictions using trained model",
                required_inputs=[
                    "data/features/realtime/*.parquet",
                    "models/catboost/model.cbm"
                ],
                expected_outputs=["models/catboost/predictions_realtime.csv"],
                agent_function=self._run_prediction
            ),
            RealtimePipelineStage(
                name="deployment",
                description="Push predictions to Vertica production table",
                required_inputs=["models/catboost/predictions_realtime.csv"],
                expected_outputs=[],
                agent_function=self._run_deployment,
                enabled=True  # Always enabled for real-time (unlike training pipeline)
            )
        ]

    def _validate_inputs(self, stage: RealtimePipelineStage) -> List[str]:
        """Validate that required input files exist."""
        missing = []
        for pattern in stage.required_inputs:
            path = Path(pattern)
            
            if "*" in pattern:
                # Handle glob patterns
                parent = path.parent
                if not list(parent.glob(path.name)):
                    missing.append(pattern)
            else:
                # Handle specific files
                if not path.exists():
                    missing.append(pattern)
        return missing

    def _run_ingestion(self) -> RealtimePipelineResult:
        """Execute ingestion agent for real-time data."""
        start_time = time.time()
        try:
            # Ensure we're in realtime mode
            original_mode = settings.INGEST_MODE
            settings.INGEST_MODE = 'realtime'
            
            output_path = ingestion_agent.ingest_once()
            execution_time = time.time() - start_time
            
            # Restore original mode
            settings.INGEST_MODE = original_mode
            
            return RealtimePipelineResult(
                stage="ingestion",
                success=True,
                execution_time=execution_time,
                input_files=[],
                output_files=[str(output_path)],
                metrics={"mode": "realtime"}
            )
        except Exception as e:
            return RealtimePipelineResult(
                stage="ingestion",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_preprocessing(self) -> RealtimePipelineResult:
        """Execute preprocessing agent for real-time data."""
        start_time = time.time()
        try:
            # Ensure we're in realtime mode
            original_mode = settings.INGEST_MODE
            settings.INGEST_MODE = 'realtime'
            
            output_path = preprocessing_agent.run_preprocessing()
            execution_time = time.time() - start_time
            
            # Restore original mode
            settings.INGEST_MODE = original_mode
            
            return RealtimePipelineResult(
                stage="preprocessing",
                success=True,
                execution_time=execution_time,
                input_files=["data/input_raw/realtime/*.csv"],
                output_files=[str(output_path)],
                metrics={}
            )
        except Exception as e:
            return RealtimePipelineResult(
                stage="preprocessing",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_feature_engineering(self) -> RealtimePipelineResult:
        """Execute feature engineering agent for real-time data."""
        start_time = time.time()
        try:
            # Ensure we're in realtime mode and keep ts column to match training schema
            original_mode = settings.INGEST_MODE
            original_keep_ts = getattr(settings, 'FE_KEEP_TS', 'false')
            
            settings.INGEST_MODE = 'realtime'
            settings.FE_KEEP_TS = 'true'  # Keep ts column to match training feature schema
            
            output_path = feature_engineering_agent.run_feature_engineering()
            execution_time = time.time() - start_time
            
            # Restore original settings
            settings.INGEST_MODE = original_mode
            settings.FE_KEEP_TS = original_keep_ts
            
            return RealtimePipelineResult(
                stage="feature_engineering",
                success=True,
                execution_time=execution_time,
                input_files=["data/preprocessed/realtime/*.csv"],
                output_files=[str(output_path)],
                metrics={}
            )
        except Exception as e:
            return RealtimePipelineResult(
                stage="feature_engineering",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_prediction(self) -> RealtimePipelineResult:
        """Generate predictions using the trained model."""
        start_time = time.time()
        try:
            # Load the trained model
            model_path = Path(settings.MODEL_DIR) / "model.cbm"
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {model_path}")
            
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            
            # Find the latest feature file
            feature_dir = Path(settings.FE_OUTPUT_DIR) / "realtime"
            feature_files = list(feature_dir.glob("*.parquet"))
            if not feature_files:
                feature_files = list(feature_dir.glob("*.csv"))
            
            if not feature_files:
                raise FileNotFoundError(f"No feature files found in {feature_dir}")
            
            # Use the most recent feature file
            latest_feature_file = max(feature_files, key=lambda f: f.stat().st_mtime)
            
            # Load features
            if latest_feature_file.suffix.lower() == '.parquet':
                df_features = pd.read_parquet(latest_feature_file)
            else:
                df_features = pd.read_csv(latest_feature_file)
            
            logger.info(f"Loaded {len(df_features):,} rows for prediction from {latest_feature_file}")
            
            # Prepare features for prediction using the SAME logic as training agent
            import agents.training_agent as training_agent
            
            # Use the exact same feature preparation as in training
            X, _, cat_idx, _ = training_agent._prepare_xy(df_features)
            
            logger.info(f"Prepared {len(X.columns)} features for prediction (after _prepare_xy)")
            logger.info(f"Feature columns: {list(X.columns)}")
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Round predictions if configured
            if settings.PRED_ROUND:
                predictions = pd.Series(predictions).clip(lower=0).round().astype(int).values
            
            # Create prediction output with future timestamps
            move_date_pred, move_hour_pred = training_agent._shift_move_datetime(df_features, settings.FE_HORIZON_HOURS)
            
            pred_df = pd.DataFrame({
                "TerminalID": df_features.get("TerminalID"),
                "MoveType": df_features.get("MoveType"),
                "Desig": df_features.get("Desig"),
                "MoveDate_pred": move_date_pred,
                "MoveHour_pred": move_hour_pred,
                "TokenCount_pred": predictions,
            })
            
            # Save predictions
            from utils.io import get_output_path
            pred_path = get_output_path(
                base_dir=Path(settings.MODEL_DIR),
                filename="predictions_realtime.csv",
                replace_files=settings.REPLACE_INTERMEDIATE_FILES,
                keep_last_n=settings.KEEP_LAST_N_VERSIONS,
                cleanup_pattern="predictions_realtime*.csv"
            )
            pred_df.to_csv(pred_path, index=False)
            
            execution_time = time.time() - start_time
            
            return RealtimePipelineResult(
                stage="prediction",
                success=True,
                execution_time=execution_time,
                input_files=[str(latest_feature_file), str(model_path)],
                output_files=[str(pred_path)],
                metrics={
                    "predictions_generated": len(predictions),
                    "model_used": str(model_path),
                    "features_from": str(latest_feature_file)
                }
            )
        except Exception as e:
            return RealtimePipelineResult(
                stage="prediction",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_deployment(self) -> RealtimePipelineResult:
        """Push real-time predictions to Vertica."""
        start_time = time.time()
        try:
            pred_path = Path(settings.MODEL_DIR) / "predictions_realtime.csv"
            if not pred_path.exists():
                raise FileNotFoundError(f"Predictions file not found: {pred_path}")
            
            rows_pushed = vertica_push_agent.push_predictions(pred_path)
            execution_time = time.time() - start_time
            
            return RealtimePipelineResult(
                stage="deployment",
                success=True,
                execution_time=execution_time,
                input_files=[str(pred_path)],
                output_files=[],
                metrics={"rows_pushed": rows_pushed}
            )
        except Exception as e:
            return RealtimePipelineResult(
                stage="deployment",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def run_stage(self, stage_name: str) -> RealtimePipelineResult:
        """Run a specific pipeline stage."""
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if not stage:
            raise ValueError(f"Unknown stage: {stage_name}")
            
        if not stage.enabled:
            logger.warning(f"Stage '{stage_name}' is disabled. Enable it first.")
            return RealtimePipelineResult(
                stage=stage_name,
                success=False,
                execution_time=0,
                input_files=[],
                output_files=[],
                metrics={},
                error_message="Stage disabled"
            )
            
        logger.info(f"🚀 Starting real-time stage: {stage.name} - {stage.description}")
        
        # Validate inputs (except for ingestion which has no inputs)
        if stage.name != "ingestion":
            missing_inputs = self._validate_inputs(stage)
            if missing_inputs:
                error_msg = f"Missing required inputs: {missing_inputs}"
                logger.error(error_msg)
                return RealtimePipelineResult(
                    stage=stage_name,
                    success=False,
                    execution_time=0,
                    input_files=[],
                    output_files=[],
                    metrics={},
                    error_message=error_msg
                )
        
        # Execute stage
        result = stage.agent_function()
        self.results.append(result)
        
        if result.success:
            logger.info(f"✅ Stage '{stage.name}' completed in {result.execution_time:.2f}s")
        else:
            logger.error(f"❌ Stage '{stage.name}' failed: {result.error_message}")
            
        return result

    def run_full_pipeline(self, stop_on_error: bool = True) -> Dict[str, Any]:
        """Run the complete real-time prediction pipeline."""
        logger.info("=" * 80)
        logger.info("🔄 STARTING REAL-TIME PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        failed_stages = []
        
        for stage in self.stages:
            if not stage.enabled:
                logger.info(f"⏭️ Skipping disabled stage: {stage.name}")
                continue
                
            result = self.run_stage(stage.name)
            
            if not result.success:
                failed_stages.append(stage.name)
                if stop_on_error:
                    logger.error(f"⚠️ Stopping pipeline due to failure in: {stage.name}")
                    break
        
        total_time = time.time() - self.start_time.timestamp()
        
        # Generate summary
        summary = {
            "pipeline_type": "realtime_prediction",
            "start_time": self.start_time.isoformat(),
            "total_execution_time": total_time,
            "stages_executed": len(self.results),
            "stages_successful": len([r for r in self.results if r.success]),
            "stages_failed": failed_stages,
            "overall_success": len(failed_stages) == 0,
            "detailed_results": [
                {
                    "stage": r.stage,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "metrics": r.metrics,
                    "error": r.error_message
                } for r in self.results
            ]
        }
        
        # Save pipeline report
        report_path = Path("data/_reports/pipeline") / f"realtime_pipeline_report_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Log summary
        logger.info("=" * 80)
        if summary["overall_success"]:
            logger.info(f"🎉 REAL-TIME PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}s")
            if self.results and self.results[-1].stage == "deployment":
                rows_pushed = self.results[-1].metrics.get("rows_pushed", 0)
                logger.info(f"📊 Pushed {rows_pushed} predictions to Vertica")
        else:
            logger.error(f"💥 REAL-TIME PIPELINE FAILED. Failed stages: {failed_stages}")
        logger.info(f"📊 Report saved: {report_path}")
        logger.info("=" * 80)
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Prediction Pipeline Orchestrator")
    parser.add_argument("--stage", type=str, help="Run specific stage only")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue pipeline even if a stage fails")
    
    args = parser.parse_args()
    
    orchestrator = RealtimePipelineOrchestrator()
    
    try:
        if args.stage:
            # Run specific stage
            result = orchestrator.run_stage(args.stage)
            sys.exit(0 if result.success else 1)
        else:
            # Run full pipeline
            summary = orchestrator.run_full_pipeline(stop_on_error=not args.continue_on_error)
            sys.exit(0 if summary["overall_success"] else 1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Real-time pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
