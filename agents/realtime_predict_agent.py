#!/usr/bin/env python3
"""
Realtime Prediction Agent - Watch for new data and make live predictions
"""

import pandas as pd
import numpy as np
import logging
import yaml
import joblib
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import threading
import argparse
from typing import Dict, Any, Optional, Tuple

# Import pipeline agents
from agents.preprocessing_agent import PreprocessingAgent
from agents.aggregation_agent import AggregationAgent
from agents.feature_agent import FeatureAgent
from agents.scaling_agent import ScalingAgent

class RealtimePredictAgent:
    def __init__(self, config_path: str = "config.yaml", log_dir: str = "logs"):
        self.config_path = config_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "realtime_predict_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RealtimePredictAgent')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.staging_dir = Path("data/staging")
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        
        self.predictions_output_dir = Path("data/realtime_predictions")
        self.predictions_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize pipeline agents
        self.preprocessing_agent = PreprocessingAgent(log_dir)
        self.aggregation_agent = AggregationAgent(log_dir)
        self.feature_agent = FeatureAgent(log_dir)
        self.scaling_agent = ScalingAgent(log_dir)
        
        # Champion model cache
        self._champion_model = None
        self._champion_metadata = None
        self._scaler = None
        self._autoencoder = None
        
        # File watcher
        self.observer = None
        self.is_running = False
        
        self.logger.info("Realtime Prediction Agent initialized")
    
    def load_champion_model(self) -> bool:
        """Load the current champion model and its metadata"""
        try:
            champion_file = Path(self.config['models']['champion_file'])
            
            if not champion_file.exists():
                self.logger.error("No champion model selected. Please set a champion first.")
                return False
            
            # Read champion model name
            with open(champion_file, 'r') as f:
                champion_name = f.read().strip()
            
            self.logger.info(f"Loading champion model: {champion_name}")
            
            # Find latest metadata file for this model
            predictions_dir = Path(self.config['data']['predictions_dir']) / champion_name
            metadata_files = list(predictions_dir.glob("*_metadata_*.yaml"))
            
            if not metadata_files:
                self.logger.error(f"No metadata found for champion model: {champion_name}")
                return False
            
            # Get latest metadata
            latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            # Load metadata with custom YAML handling for NumPy objects
            try:
                with open(latest_metadata_file, 'r') as f:
                    # Try safe_load first
                    content = f.read()
                    # Remove problematic NumPy object serializations for basic info
                    lines = content.split('\n')
                    filtered_lines = []
                    skip_lines = False
                    
                    for line in lines:
                        if 'test_metrics:' in line:
                            skip_lines = True
                            continue
                        if skip_lines and (line.startswith('  ') or line.strip() == ''):
                            continue
                        if skip_lines and not line.startswith(' '):
                            skip_lines = False
                        
                        if not skip_lines:
                            filtered_lines.append(line)
                    
                    filtered_content = '\n'.join(filtered_lines)
                    self._champion_metadata = yaml.safe_load(filtered_content)
            except Exception as e:
                self.logger.error(f"Error parsing metadata file {latest_metadata_file}: {e}")
                return False
            
            # Load the model
            model_path = self._champion_metadata['model_path']
            self._champion_model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = Path("models/scaler.pkl")
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
            else:
                self.logger.error("Scaler not found. Pipeline may not be complete.")
                return False
            
            # Load autoencoder if model uses dense embeddings
            if self._champion_metadata['data_type'] == 'dense':
                autoencoder_path = Path("models/autoencoder.h5")
                if autoencoder_path.exists():
                    import tensorflow as tf
                    self._autoencoder = tf.keras.models.load_model(autoencoder_path)
                else:
                    self.logger.error("Autoencoder not found for dense model. Pipeline may not be complete.")
                    return False
            
            self.logger.info(f"Successfully loaded champion model: {champion_name}")
            self.logger.info(f"Model type: {self._champion_metadata['data_type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading champion model: {str(e)}")
            return False
    
    def process_new_data(self, input_file: Path) -> Optional[Path]:
        """Process new staging data through the pipeline"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = Path(f"data/temp_realtime_{timestamp}")
            temp_dir.mkdir(exist_ok=True, parents=True)
            
            self.logger.info(f"Processing new data: {input_file}")
            
            # Step 1: Preprocessing
            raw_file = temp_dir / "moves_raw.csv"
            clean_file = temp_dir / "moves_clean.csv"
            
            # Convert Excel to CSV if needed
            if input_file.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(input_file)
                df.to_csv(raw_file, index=False)
            else:
                # Copy CSV file
                import shutil
                shutil.copy2(input_file, raw_file)
            
            success = self.preprocessing_agent.run(
                input_path=str(raw_file),
                output_path=str(clean_file)
            )
            if not success:
                self.logger.error("Preprocessing failed")
                return None
            
            # Step 2: Aggregation
            wide_file = temp_dir / "moves_wide.csv"
            success = self.aggregation_agent.run(
                input_path=str(clean_file),
                output_path=str(wide_file)
            )
            if not success:
                self.logger.error("Aggregation failed")
                return None
            
            # Step 3: Feature Engineering
            features_file = temp_dir / "moves_features.csv"
            success = self.feature_agent.run(
                input_path=str(wide_file),
                output_path=str(features_file)
            )
            if not success:
                self.logger.error("Feature engineering failed")
                return None
            
            # Step 4: Scaling
            scaled_file = temp_dir / "moves_scaled.csv"
            success = self._scale_data(features_file, scaled_file)
            if not success:
                self.logger.error("Scaling failed")
                return None
            
            # Step 5: Encoding (if needed for dense models)
            if self._champion_metadata['data_type'] == 'dense':
                encoded_file = temp_dir / "moves_encoded.csv"
                success = self._encode_data(scaled_file, encoded_file)
                if not success:
                    self.logger.error("Encoding failed")
                    return None
                prediction_input_file = encoded_file
            else:
                prediction_input_file = scaled_file
            
            self.logger.info("Data processing pipeline completed successfully")
            return prediction_input_file
            
        except Exception as e:
            self.logger.error(f"Error processing new data: {str(e)}")
            return None
    
    def _scale_data(self, features_file: Path, output_file: Path) -> bool:
        """Scale features using pre-fitted scaler"""
        try:
            # Load features
            df = pd.read_csv(features_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Separate timestamp from features
            timestamps = df['timestamp']
            feature_cols = [col for col in df.columns if col != 'timestamp']
            features = df[feature_cols]
            
            # Scale features
            scaled_features = self._scaler.transform(features)
            
            # Create scaled dataframe
            scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
            scaled_df.insert(0, 'timestamp', timestamps)
            
            # Save
            scaled_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully scaled {len(scaled_df)} rows")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scaling data: {str(e)}")
            return False
    
    def _encode_data(self, scaled_file: Path, output_file: Path) -> bool:
        """Encode data using pre-trained autoencoder"""
        try:
            import tensorflow as tf
            
            # Load scaled data
            df = pd.read_csv(scaled_file)
            timestamps = df['timestamp']
            
            # Get features (exclude timestamp)
            features = df.drop('timestamp', axis=1)
            
            # Get embeddings from encoder
            encoder = tf.keras.Model(inputs=self._autoencoder.input, 
                                   outputs=self._autoencoder.get_layer('encoder_output').output)
            embeddings = encoder.predict(features.values, verbose=0)
            
            # Create embedded dataframe
            embed_cols = [f'embed_{i}' for i in range(embeddings.shape[1])]
            embed_df = pd.DataFrame(embeddings, columns=embed_cols)
            embed_df.insert(0, 'timestamp', timestamps)
            
            # Save
            embed_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Successfully encoded {len(embed_df)} rows to {embeddings.shape[1]} dimensions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error encoding data: {str(e)}")
            return False
    
    def make_predictions(self, input_file: Path) -> Optional[Path]:
        """Make predictions using the champion model"""
        try:
            # Load processed data
            df = pd.read_csv(input_file)
            timestamps = df['timestamp']
            
            # Get features (exclude timestamp)
            features = df.drop('timestamp', axis=1)
            
            # Make predictions
            predictions = self._champion_model.predict(features)
            
            # Create predictions dataframe
            pred_df = pd.DataFrame({
                'timestamp': timestamps,
                'predicted_count': predictions
            })
            
            # Save predictions with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.predictions_output_dir / f"realtime_predictions_{timestamp}.csv"
            pred_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Generated {len(predictions)} predictions")
            self.logger.info(f"Predictions saved to: {output_file}")
            
            # Also save latest predictions
            latest_file = self.predictions_output_dir / "latest_predictions.csv"
            pred_df.to_csv(latest_file, index=False)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary processing files older than max_age_hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for temp_dir in Path("data").glob("temp_realtime_*"):
                if temp_dir.is_dir():
                    dir_time = datetime.fromtimestamp(temp_dir.stat().st_mtime)
                    if dir_time < cutoff_time:
                        import shutil
                        shutil.rmtree(temp_dir)
                        self.logger.info(f"Cleaned up temp directory: {temp_dir}")
                        
        except Exception as e:
            self.logger.error(f"Error cleaning temp files: {str(e)}")

class StagingFileHandler(FileSystemEventHandler):
    """Handle new files in staging directory"""
    
    def __init__(self, predict_agent: RealtimePredictAgent):
        self.predict_agent = predict_agent
        self.logger = predict_agent.logger
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process Excel and CSV files
        if file_path.suffix.lower() not in ['.xlsx', '.xls', '.csv']:
            return
        
        self.logger.info(f"New file detected: {file_path}")
        
        # Wait a moment for file to be fully written
        time.sleep(2)
        
        # Process the file
        self._process_file(file_path)
    
    def _process_file(self, file_path: Path):
        """Process a single file"""
        try:
            # Process through pipeline
            processed_file = self.predict_agent.process_new_data(file_path)
            
            if processed_file:
                # Make predictions
                predictions_file = self.predict_agent.make_predictions(processed_file)
                
                if predictions_file:
                    self.logger.info(f"✅ Successfully processed {file_path.name}")
                    
                    # Move processed file to archive
                    archive_dir = Path("data/staging/processed")
                    archive_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_file = archive_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                    
                    import shutil
                    shutil.move(file_path, archive_file)
                    self.logger.info(f"Archived input file to: {archive_file}")
                else:
                    self.logger.error(f"❌ Failed to generate predictions for {file_path.name}")
            else:
                self.logger.error(f"❌ Failed to process {file_path.name}")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")

def start_streamlit_dashboard():
    """Start Streamlit dashboard for realtime predictions"""
    try:
        # Create a simple realtime dashboard
        dashboard_path = Path("streamlit_realtime_predictions.py")
        
        if not dashboard_path.exists():
            dashboard_content = '''#!/usr/bin/env python3
"""
Streamlit Dashboard for Realtime Predictions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Realtime Predictions Dashboard",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Realtime Predictions Dashboard")

# Auto-refresh
if st.button("🔄 Refresh"):
    st.rerun()

# Auto-refresh every 30 seconds
time.sleep(0.1)  # Small delay to prevent too frequent updates

# Load latest predictions
predictions_dir = Path("data/realtime_predictions")
latest_file = predictions_dir / "latest_predictions.csv"

if latest_file.exists():
    df = pd.read_csv(latest_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    st.subheader("📊 Latest Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['predicted_count'],
            mode='lines+markers',
            name='Predicted Count',
            line=dict(color='#1f77b4')
        ))
        
        fig.update_layout(
            title="Prediction Timeline",
            xaxis_title="Time",
            yaxis_title="Predicted Count",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary stats
        st.metric("Total Predictions", len(df))
        st.metric("Latest Prediction", f"{df['predicted_count'].iloc[-1]:.2f}")
        st.metric("Average", f"{df['predicted_count'].mean():.2f}")
        st.metric("Max Prediction", f"{df['predicted_count'].max():.2f}")
    
    # Recent predictions table
    st.subheader("📋 Recent Predictions")
    recent_df = df.tail(10).copy()
    recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    recent_df['predicted_count'] = recent_df['predicted_count'].round(2)
    st.dataframe(recent_df, use_container_width=True)
    
else:
    st.info("No predictions available yet. Place data files in the staging directory to generate predictions.")

# Show all prediction files
st.subheader("📁 Prediction History")
if predictions_dir.exists():
    pred_files = list(predictions_dir.glob("realtime_predictions_*.csv"))
    if pred_files:
        file_info = []
        for file in pred_files:
            stat = file.stat()
            file_info.append({
                'File': file.name,
                'Size': f"{stat.st_size / 1024:.1f} KB",
                'Modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        files_df = pd.DataFrame(file_info)
        st.dataframe(files_df, use_container_width=True)
    else:
        st.info("No prediction files found.")
'''
            
            with open(dashboard_path, 'w') as f:
                f.write(dashboard_content)
        
        # Start dashboard
        subprocess.Popen([
            "streamlit", "run", str(dashboard_path),
            "--server.port", "8503",
            "--server.headless", "true"
        ])
        
        return "http://localhost:8503"
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return None

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Realtime Prediction Agent')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--staging-dir', default='data/staging', help='Staging directory to watch')
    parser.add_argument('--dashboard', action='store_true', help='Start Streamlit dashboard')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old temp files and exit')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RealtimePredictAgent(config_path=args.config)
    
    if args.cleanup:
        agent.cleanup_temp_files()
        return
    
    # Load champion model
    if not agent.load_champion_model():
        print("❌ Failed to load champion model. Please ensure a champion is selected.")
        return
    
    # Start dashboard if requested
    dashboard_url = None
    if args.dashboard:
        dashboard_url = start_streamlit_dashboard()
        if dashboard_url:
            agent.logger.info(f"Streamlit dashboard started at: {dashboard_url}")
        else:
            agent.logger.error("Failed to start Streamlit dashboard")
    
    # Setup file watcher
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(exist_ok=True)
    
    event_handler = StagingFileHandler(agent)
    agent.observer = Observer()
    agent.observer.schedule(event_handler, str(staging_dir), recursive=False)
    
    # Start watching
    agent.observer.start()
    agent.is_running = True
    
    agent.logger.info("🚀 Realtime Prediction Agent started")
    agent.logger.info(f"👀 Watching directory: {staging_dir.absolute()}")
    agent.logger.info(f"🏆 Champion model: {agent._champion_metadata['model']}")
    if dashboard_url:
        agent.logger.info(f"📊 Dashboard: {dashboard_url}")
    agent.logger.info("📝 Drop Excel/CSV files in staging directory to generate predictions")
    agent.logger.info("Press Ctrl+C to stop")
    
    try:
        # Cleanup task every hour
        last_cleanup = time.time()
        
        while agent.is_running:
            time.sleep(10)
            
            # Periodic cleanup
            if time.time() - last_cleanup > 3600:  # 1 hour
                agent.cleanup_temp_files()
                last_cleanup = time.time()
                
    except KeyboardInterrupt:
        agent.logger.info("Stopping Realtime Prediction Agent...")
        agent.observer.stop()
        agent.is_running = False
    
    agent.observer.join()
    agent.logger.info("Realtime Prediction Agent stopped")

if __name__ == "__main__":
    main()
