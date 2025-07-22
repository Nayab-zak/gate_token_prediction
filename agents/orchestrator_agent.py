#!/usr/bin/env python3
"""
Orchestrator Agent - Main pipeline controller
"""

import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime

# Import all agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.aggregation_agent import AggregationAgent
from agents.feature_agent import FeatureAgent
from agents.scaling_agent import ScalingAgent
from agents.encoder_agent import EncoderAgent

class OrchestratorAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "orchestrator_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OrchestratorAgent')
        
        # Define pipeline steps
        self.steps = [
            'ingestion',
            'preprocessing', 
            'aggregation',
            'feature',
            'scaling',
            'encoder'
        ]
        
        # Track completion
        self.completed_steps = set()
    
    def run_ingestion(self):
        """Run ingestion step"""
        try:
            self.logger.info("=== STEP 1: INGESTION ===")
            
            agent = IngestionAgent(self.log_dir)
            success = agent.run(
                input_path="data/input/moves.xlsx",
                output_path="data/preprocessed/moves_raw.csv"
            )
            
            if success:
                self.completed_steps.add('ingestion')
                self.logger.info("✓ Ingestion completed successfully")
                return True
            else:
                self.logger.error("✗ Ingestion failed")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Ingestion failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_preprocessing(self):
        """Run preprocessing step"""
        try:
            self.logger.info("=== STEP 2: PREPROCESSING ===")
            
            agent = PreprocessingAgent(self.log_dir)
            success = agent.run(
                input_path="data/preprocessed/moves_raw.csv",
                output_path="data/preprocessed/moves_clean.csv",
                correlation_threshold=0.89
            )
            
            if success:
                self.completed_steps.add('preprocessing')
                self.logger.info("✓ Preprocessing completed successfully")
                return True
            else:
                self.logger.error("✗ Preprocessing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Preprocessing failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_aggregation(self):
        """Run aggregation step"""
        try:
            self.logger.info("=== STEP 3: AGGREGATION ===")
            
            agent = AggregationAgent(self.log_dir)
            success = agent.run(
                input_path="data/preprocessed/moves_clean.csv",
                output_path="data/preprocessed/moves_wide.csv"
            )
            
            if success:
                self.completed_steps.add('aggregation')
                self.logger.info("✓ Aggregation completed successfully")
                return True
            else:
                self.logger.error("✗ Aggregation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Aggregation failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_feature(self):
        """Run feature engineering step"""
        try:
            self.logger.info("=== STEP 4: FEATURE ENGINEERING ===")
            
            agent = FeatureAgent(self.log_dir)
            success = agent.run(
                input_path="data/preprocessed/moves_wide.csv",
                output_path="data/preprocessed/moves_features.csv",
                lag_hours=[1, 24, 168],
                windows=[3, 24, 168]
            )
            
            if success:
                self.completed_steps.add('feature')
                self.logger.info("✓ Feature engineering completed successfully")
                return True
            else:
                self.logger.error("✗ Feature engineering failed")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Feature engineering failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_scaling(self):
        """Run scaling step"""
        try:
            self.logger.info("=== STEP 5: SCALING ===")
            
            agent = ScalingAgent(self.log_dir)
            success = agent.run(
                input_path="data/preprocessed/moves_features.csv",
                output_dir="data/preprocessed",
                scaler_path="models/scaler.pkl",
                train_ratio=0.7,
                val_ratio=0.15
            )
            
            if success:
                self.completed_steps.add('scaling')
                self.logger.info("✓ Scaling completed successfully")
                return True
            else:
                self.logger.error("✗ Scaling failed")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Scaling failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_encoder(self):
        """Run encoder step"""
        try:
            self.logger.info("=== STEP 6: ENCODER ===")
            
            agent = EncoderAgent(self.log_dir)
            success = agent.run(
                train_path="data/preprocessed/X_train_scaled.csv",
                val_path="data/preprocessed/X_val_scaled.csv",
                test_path="data/preprocessed/X_test_scaled.csv",
                model_path="models/autoencoder.h5",
                output_dir="data/encoded_input",
                encoding_dim=64,
                epochs=100
            )
            
            if success:
                self.completed_steps.add('encoder')
                self.logger.info("✓ Encoder training completed successfully")
                return True
            else:
                self.logger.error("✗ Encoder training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Encoder training failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_full_pipeline(self, start_from_step=1):
        """Run the complete pipeline or resume from a specific step"""
        try:
            start_time = datetime.now()
            self.logger.info(f"🚀 Starting pipeline execution from step {start_from_step}")
            self.logger.info(f"Start time: {start_time}")
            
            # Map step numbers to methods
            step_methods = {
                1: self.run_ingestion,
                2: self.run_preprocessing,
                3: self.run_aggregation, 
                4: self.run_feature,
                5: self.run_scaling,
                6: self.run_encoder
            }
            
            # Run steps from start_from_step onwards
            for step_num in range(start_from_step, 7):
                step_method = step_methods[step_num]
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Starting step {step_num}")
                self.logger.info(f"{'='*60}")
                
                success = step_method()
                
                if not success:
                    self.logger.error(f"💥 Pipeline failed at step {step_num}")
                    return False
                
                self.logger.info(f"✅ Step {step_num} completed")
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Start time: {start_time}")
            self.logger.info(f"End time: {end_time}")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info(f"Completed steps: {sorted(list(self.completed_steps))}")
            self.logger.info("\n📁 Output files created:")
            self.logger.info("  - data/encoded_input/Z_train.csv")
            self.logger.info("  - data/encoded_input/Z_val.csv") 
            self.logger.info("  - data/encoded_input/Z_test.csv")
            self.logger.info("  - models/autoencoder.h5")
            self.logger.info("  - models/scaler.pkl")
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Pipeline failed with exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_pipeline_status(self):
        """Get current pipeline status"""
        status = {
            'completed_steps': sorted(list(self.completed_steps)),
            'remaining_steps': [step for step in self.steps if step not in self.completed_steps],
            'progress': f"{len(self.completed_steps)}/{len(self.steps)}"
        }
        return status

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Orchestrator Agent - ML Pipeline Controller')
    parser.add_argument('--start-from', type=int, default=1, choices=range(1, 7),
                       help='Step number to start from (1=ingestion, 2=preprocessing, 3=aggregation, 4=feature, 5=scaling, 6=encoder)')
    parser.add_argument('--log-dir', default='logs', help='Directory for log files')
    
    args = parser.parse_args()
    
    # Create orchestrator and run pipeline
    orchestrator = OrchestratorAgent(args.log_dir)
    success = orchestrator.run_full_pipeline(args.start_from)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
