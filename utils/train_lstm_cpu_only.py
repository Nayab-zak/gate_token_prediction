import os
# Force CPU only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_lstm_cpu_only')

def run_command(cmd):
    logger.info(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        logger.error(f"Error output: {stderr}")
        return False
    
    logger.info(f"Command output: {stdout}")
    return True

if __name__ == '__main__':
    logger.info("Starting LSTM training with CPU-only mode")
    
    # Train LSTM Classic model
    logger.info("Training LSTM Classic model")
    success = run_command("python agents/06_train_agents/06_train_lstm_classic_agent.py")
    
    if success:
        # Train LSTM Augmented model
        logger.info("Training LSTM Augmented model")
        success = run_command("python agents/06_train_agents/06_train_lstm_augmented_agent.py")
    
    if success:
        logger.info("✅ LSTM training completed successfully in CPU-only mode")
    else:
        logger.error("❌ LSTM training failed")
        sys.exit(1)
