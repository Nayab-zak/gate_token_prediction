#!/usr/bin/env python3
"""
Pipeline Orchestrator for Gate Token Prediction
===============================================

This orchestrator manages the complete ML pipeline while keeping individual agents modular.
Benefits:
- Centralized error handling and logging
- Data validation between stages  
- Pipeline state management
- Flexible execution (full pipeline or individual stages)
- Performance monitoring and metrics
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

from config import settings
from utils.logging import get_logger

# Import individual agents
from agents import (
    ingestion_agent,
    preprocessing_agent, 
    split_agent,
    training_agent,
    evaluate_agent,
    vertica_push_agent
)

logger = get_logger(__name__)

@dataclass
class PipelineStage:
    name: str
    description: str
    required_inputs: List[str]
    expected_outputs: List[str]
    agent_function: callable
    enabled: bool = True
    
@dataclass 
class PipelineResult:
    stage: str
    success: bool
    execution_time: float
    input_files: List[str]
    output_files: List[str]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class PipelineOrchestrator:
    """
    Main pipeline orchestrator that coordinates all ML agents.
    Maintains modularity while providing centralized control.
    """
    
    def __init__(self, mode: str = "history"):
        self.mode = mode
        self.results: List[PipelineResult] = []
        self.start_time = datetime.now()
        
        # Define pipeline stages
        self.stages = [
            PipelineStage(
                name="ingestion",
                description="Extract data from Vertica database", 
                required_inputs=[],
                expected_outputs=["data/input_raw/{mode}/*.csv"],
                agent_function=self._run_ingestion
            ),
            PipelineStage(
                name="preprocessing", 
                description="Clean and standardize raw data",
                required_inputs=["data/input_raw/{mode}/*.csv"],
                expected_outputs=["data/preprocessed/{mode}/*.csv"],
                agent_function=self._run_preprocessing
            ),
            PipelineStage(
                name="splitting",
                description="Split data and engineer features (prevents leakage)",
                required_inputs=["data/preprocessed/{mode}/*.csv"], 
                expected_outputs=[
                    "data/features/{mode}/features_train.parquet",
                    "data/features/{mode}/features_valid.parquet", 
                    "data/features/{mode}/features_test.parquet"
                ],
                agent_function=self._run_splitting
            ),
            PipelineStage(
                name="training",
                description="Train CatBoost model and generate predictions",
                required_inputs=[
                    "data/features/{mode}/features_train.parquet",
                    "data/features/{mode}/features_valid.parquet",
                    "data/features/{mode}/features_test.parquet"
                ],
                expected_outputs=[
                    "models/catboost/model.cbm",
                    "models/catboost/predictions_test.csv"
                ],
                agent_function=self._run_training
            ),
            PipelineStage(
                name="evaluation", 
                description="Generate evaluation reports and visualizations",
                required_inputs=["models/catboost/predictions_test.csv"],
                expected_outputs=["data/_reports/eval/metrics_test.json"],
                agent_function=self._run_evaluation
            ),
            PipelineStage(
                name="deployment",
                description="Push predictions to Vertica database", 
                required_inputs=["models/catboost/predictions_test.csv"],
                expected_outputs=[],
                agent_function=self._run_deployment,
                enabled=False  # Manual enable for production deployment
            )
        ]

    def _validate_inputs(self, stage: PipelineStage) -> List[str]:
        """Validate that required input files exist."""
        missing = []
        for pattern in stage.required_inputs:
            resolved_pattern = pattern.format(mode=self.mode)
            path = Path(resolved_pattern)
            
            if "*" in resolved_pattern:
                # Handle glob patterns
                parent = path.parent
                if not list(parent.glob(path.name)):
                    missing.append(resolved_pattern)
            else:
                # Handle specific files
                if not path.exists():
                    missing.append(resolved_pattern)
        return missing

    def _run_ingestion(self) -> PipelineResult:
        """Execute ingestion agent."""
        start_time = time.time()
        try:
            output_path = ingestion_agent.ingest_once()
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage="ingestion",
                success=True,
                execution_time=execution_time,
                input_files=[],
                output_files=[str(output_path)],
                metrics={"rows_extracted": "unknown"}  # Could enhance agents to return metrics
            )
        except Exception as e:
            return PipelineResult(
                stage="ingestion", 
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_preprocessing(self) -> PipelineResult:
        """Execute preprocessing agent."""
        start_time = time.time()
        try:
            output_path = preprocessing_agent.run_preprocessing()
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage="preprocessing",
                success=True, 
                execution_time=execution_time,
                input_files=["data/input_raw/{mode}/*.csv".format(mode=self.mode)],
                output_files=[str(output_path)],
                metrics={}
            )
        except Exception as e:
            return PipelineResult(
                stage="preprocessing",
                success=False,
                execution_time=time.time() - start_time, 
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_splitting(self) -> PipelineResult:
        """Execute split agent (includes feature engineering)."""
        start_time = time.time()
        try:
            output_paths = split_agent.run_data_splitting()
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage="splitting",
                success=True,
                execution_time=execution_time, 
                input_files=["data/preprocessed/{mode}/*.csv".format(mode=self.mode)],
                output_files=[str(p) for p in output_paths.values()],
                metrics={"splits_created": len(output_paths)}
            )
        except Exception as e:
            return PipelineResult(
                stage="splitting",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[], 
                metrics={},
                error_message=str(e)
            )

    def _run_training(self) -> PipelineResult:
        """Execute training agent.""" 
        start_time = time.time()
        try:
            training_agent.main()
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage="training", 
                success=True,
                execution_time=execution_time,
                input_files=[
                    settings.TRAIN_TRAIN_PATH,
                    settings.TRAIN_VALID_PATH, 
                    settings.TRAIN_TEST_PATH
                ],
                output_files=[
                    f"{settings.MODEL_DIR}/model.cbm",
                    f"{settings.MODEL_DIR}/predictions_test.csv"
                ],
                metrics={}
            )
        except Exception as e:
            return PipelineResult(
                stage="training",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_evaluation(self) -> PipelineResult:
        """Execute evaluation agent."""
        start_time = time.time()
        try:
            # Temporarily clear sys.argv to prevent argument parsing conflicts
            original_argv = sys.argv[:]
            sys.argv = [sys.argv[0]]  # Keep only the script name
            
            try:
                evaluate_agent.main()
            finally:
                sys.argv = original_argv  # Restore original arguments
                
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage="evaluation",
                success=True, 
                execution_time=execution_time,
                input_files=[settings.EVAL_PRED_PATH],
                output_files=[f"{settings.EVAL_REPORT_DIR}/metrics_test.json"],
                metrics={}
            )
        except Exception as e:
            return PipelineResult(
                stage="evaluation",
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[],
                metrics={},
                error_message=str(e)
            )

    def _run_deployment(self) -> PipelineResult:
        """Execute vertica push agent."""
        start_time = time.time() 
        try:
            rows_pushed = vertica_push_agent.push_predictions(Path(settings.PREDICTIONS_PATH))
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage="deployment",
                success=True,
                execution_time=execution_time,
                input_files=[settings.PREDICTIONS_PATH],
                output_files=[],
                metrics={"rows_pushed": rows_pushed}
            )
        except Exception as e:
            return PipelineResult(
                stage="deployment", 
                success=False,
                execution_time=time.time() - start_time,
                input_files=[],
                output_files=[], 
                metrics={},
                error_message=str(e)
            )

    def run_stage(self, stage_name: str) -> PipelineResult:
        """Run a specific pipeline stage."""
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if not stage:
            raise ValueError(f"Unknown stage: {stage_name}")
            
        if not stage.enabled:
            logger.warning(f"Stage '{stage_name}' is disabled. Enable it first.")
            return PipelineResult(
                stage=stage_name,
                success=False, 
                execution_time=0,
                input_files=[],
                output_files=[],
                metrics={},
                error_message="Stage disabled"
            )
            
        logger.info(f"🚀 Starting stage: {stage.name} - {stage.description}")
        
        # Validate inputs
        missing_inputs = self._validate_inputs(stage)
        if missing_inputs:
            error_msg = f"Missing required inputs: {missing_inputs}"
            logger.error(error_msg)
            return PipelineResult(
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

    def run_full_pipeline(self, skip_on_error: bool = False) -> Dict[str, Any]:
        """Run the complete ML pipeline."""
        logger.info("=" * 80)
        logger.info(f"🔄 STARTING FULL PIPELINE - Mode: {self.mode}")
        logger.info("=" * 80)
        
        failed_stages = []
        
        for stage in self.stages:
            if not stage.enabled:
                logger.info(f"⏭️ Skipping disabled stage: {stage.name}")
                continue
                
            result = self.run_stage(stage.name)
            
            if not result.success:
                failed_stages.append(stage.name)
                if skip_on_error:
                    logger.warning(f"⚠️ Skipping remaining stages due to failure in: {stage.name}")
                    break
        
        total_time = time.time() - self.start_time.timestamp()
        
        # Generate summary
        summary = {
            "pipeline_mode": self.mode,
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
                    "error": r.error_message
                } for r in self.results
            ]
        }
        
        # Save pipeline report
        report_path = Path("data/_reports/pipeline") / f"pipeline_report_{self.mode}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Log summary
        logger.info("=" * 80)
        if summary["overall_success"]:
            logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}s")
        else:
            logger.error(f"💥 PIPELINE FAILED. Failed stages: {failed_stages}")
        logger.info(f"📊 Report saved: {report_path}")
        logger.info("=" * 80)
        
        return summary

    def enable_stage(self, stage_name: str):
        """Enable a specific stage (useful for deployment)."""
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if stage:
            stage.enabled = True
            logger.info(f"✅ Enabled stage: {stage_name}")
        else:
            logger.error(f"❌ Unknown stage: {stage_name}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["history", "realtime"], default=settings.INGEST_MODE,
                       help="Pipeline execution mode")
    parser.add_argument("--stage", type=str, help="Run specific stage only") 
    parser.add_argument("--enable-deployment", action="store_true", 
                       help="Enable deployment stage (disabled by default)")
    parser.add_argument("--skip-on-error", action="store_true",
                       help="Skip remaining stages if one fails")
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(mode=args.mode)
    
    if args.enable_deployment:
        orchestrator.enable_stage("deployment")
    
    try:
        if args.stage:
            # Run specific stage
            result = orchestrator.run_stage(args.stage)
            sys.exit(0 if result.success else 1)
        else:
            # Run full pipeline
            summary = orchestrator.run_full_pipeline(skip_on_error=args.skip_on_error)
            sys.exit(0 if summary["overall_success"] else 1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
