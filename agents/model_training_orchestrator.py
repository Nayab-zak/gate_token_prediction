#!/usr/bin/env python3
"""
Model Training Orchestrator Agent - Coordinates all model training
"""

import subprocess
import logging
import yaml
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class ModelTrainingOrchestrator:
    def __init__(self, config_path="config.yaml", log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "model_training_orchestrator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ModelTrainingOrchestrator')
        
        # Define training agents
        self.dense_models = [
            'train_random_forest.py',
            'train_extra_trees.py', 
            'train_xgboost.py',
            'train_lightgbm.py --data-type dense',
            'train_elasticnet.py',
            'train_mlp.py'
        ]
        
        self.sparse_models = [
            'train_catboost.py',
            'train_lightgbm.py --data-type sparse'
        ]
    
    def run_data_split(self):
        """Run data splitting first"""
        try:
            self.logger.info("=== RUNNING DATA SPLIT AGENT ===")
            
            cmd = [
                "python", "agents/data_split_agent.py",
                "--input-path", "data/preprocessed/moves_wide.csv",
                "--output-dir", "data/preprocessed"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✓ Data split completed successfully")
                return True
            else:
                self.logger.error(f"✗ Data split failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ Data split failed with exception: {str(e)}")
            return False
    
    def run_training_agent(self, agent_script):
        """Run a single training agent"""
        try:
            self.logger.info(f"Starting training agent: {agent_script}")
            start_time = time.time()
            
            # Parse command (handle arguments)
            cmd_parts = agent_script.split()
            cmd = ["python", f"agents/{cmd_parts[0]}"] + cmd_parts[1:]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"✓ {agent_script} completed successfully in {duration:.1f}s")
                return True, agent_script, duration
            else:
                self.logger.error(f"✗ {agent_script} failed: {result.stderr}")
                return False, agent_script, duration
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ {agent_script} timed out after 1 hour")
            return False, agent_script, 3600
        except Exception as e:
            self.logger.error(f"✗ {agent_script} failed with exception: {str(e)}")
            return False, agent_script, 0
    
    def run_all_models_sequential(self):
        """Run all models sequentially"""
        results = []
        all_models = self.dense_models + self.sparse_models
        
        for agent in all_models:
            success, agent_name, duration = self.run_training_agent(agent)
            results.append({
                'agent': agent_name,
                'success': success,
                'duration': duration
            })
        
        return results
    
    def run_all_models_parallel(self):
        """Run all models in parallel (experimental)"""
        results = []
        all_models = self.dense_models + self.sparse_models
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all jobs
            future_to_agent = {
                executor.submit(self.run_training_agent, agent): agent 
                for agent in all_models
            }
            
            # Collect results
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    success, agent_name, duration = future.result()
                    results.append({
                        'agent': agent_name,
                        'success': success,
                        'duration': duration
                    })
                except Exception as e:
                    self.logger.error(f"✗ {agent} failed with exception: {str(e)}")
                    results.append({
                        'agent': agent,
                        'success': False,
                        'duration': 0
                    })
        
        return results
    
    def start_streamlit_apps(self):
        """Start Streamlit applications"""
        try:
            self.logger.info("Starting Streamlit applications...")
            
            # Start model results app
            model_results_port = self.config['streamlit']['model_results_port']
            comparison_port = self.config['streamlit']['model_comparison_port']
            
            self.logger.info(f"To view results:")
            self.logger.info(f"  Model Results Dashboard: streamlit run streamlit_model_results.py --server.port {model_results_port}")
            self.logger.info(f"  Model Comparison Dashboard: streamlit run streamlit_model_comparison.py --server.port {comparison_port}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error with Streamlit apps: {str(e)}")
            return False
    
    def generate_summary_report(self, results):
        """Generate a summary report"""
        try:
            total_models = len(results)
            successful_models = sum(1 for r in results if r['success'])
            total_time = sum(r['duration'] for r in results)
            
            self.logger.info("="*60)
            self.logger.info("🎉 MODEL TRAINING SUMMARY REPORT")
            self.logger.info("="*60)
            self.logger.info(f"Total models attempted: {total_models}")
            self.logger.info(f"Successful models: {successful_models}")
            self.logger.info(f"Failed models: {total_models - successful_models}")
            self.logger.info(f"Total training time: {total_time/60:.1f} minutes")
            self.logger.info("")
            
            self.logger.info("📊 Model Results:")
            for result in results:
                status = "✓" if result['success'] else "✗"
                self.logger.info(f"  {status} {result['agent']:<30} ({result['duration']:.1f}s)")
            
            self.logger.info("")
            if successful_models > 0:
                self.logger.info("🔥 Next Steps:")
                self.logger.info("1. Check individual model logs in logs/ directory")
                self.logger.info("2. View predictions in data/predictions/ directory")
                self.logger.info("3. Launch Streamlit dashboards to explore results")
                
                ports = self.config['streamlit']
                self.logger.info(f"   streamlit run streamlit_model_results.py --server.port {ports['model_results_port']}")
                self.logger.info(f"   streamlit run streamlit_model_comparison.py --server.port {ports['model_comparison_port']}")
            
            # Save summary to file
            summary_path = Path("logs/training_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Model Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n")
                f.write(f"Total models: {total_models}\n")
                f.write(f"Successful: {successful_models}\n")
                f.write(f"Failed: {total_models - successful_models}\n")
                f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
                
                for result in results:
                    status = "SUCCESS" if result['success'] else "FAILED"
                    f.write(f"{result['agent']:<30} {status:<8} ({result['duration']:.1f}s)\n")
            
            self.logger.info(f"📝 Summary saved to: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
    
    def run(self):
        """Main execution method"""
        try:
            self.logger.info("🚀 Starting Model Training Orchestrator")
            self.logger.info(f"Configuration loaded from config.yaml")
            
            # Step 1: Run data split
            if not self.run_data_split():
                self.logger.error("Data split failed. Cannot continue.")
                return False
            
            # Step 2: Train all models
            parallel_training = self.config['orchestration'].get('parallel_training', False)
            
            if parallel_training:
                self.logger.info("🔄 Running model training in parallel...")
                results = self.run_all_models_parallel()
            else:
                self.logger.info("🔄 Running model training sequentially...")
                results = self.run_all_models_sequential()
            
            # Step 3: Generate summary
            self.generate_summary_report(results)
            
            # Step 4: Provide Streamlit info
            self.start_streamlit_apps()
            
            successful_models = sum(1 for r in results if r['success'])
            if successful_models > 0:
                self.logger.info("✅ Model training orchestration completed successfully!")
                return True
            else:
                self.logger.error("❌ No models trained successfully!")
                return False
                
        except Exception as e:
            self.logger.error(f"Orchestration failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Training Orchestrator')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--parallel', action='store_true', help='Run models in parallel')
    
    args = parser.parse_args()
    
    # Update config if parallel requested
    if args.parallel:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['orchestration']['parallel_training'] = True
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    orchestrator = ModelTrainingOrchestrator(args.config)
    success = orchestrator.run()
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
