#!/usr/bin/env python3
"""
Simple Model Performance Explorer - View test outputs and EDA without complex dependencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
import argparse

class ModelExplorer:
    def __init__(self, predictions_dir="data/predictions"):
        self.predictions_dir = Path(predictions_dir)
        
    def get_available_models(self):
        """Get list of available models"""
        if not self.predictions_dir.exists():
            return []
        
        models = []
        for model_dir in self.predictions_dir.iterdir():
            if model_dir.is_dir():
                # Check if there are prediction files
                pred_files = list(model_dir.glob("*_test_preds_*.csv"))
                if pred_files:
                    models.append(model_dir.name)
        
        return sorted(models)
    
    def load_model_data(self, model_name):
        """Load model predictions and metadata"""
        model_dir = self.predictions_dir / model_name
        
        # Get latest files
        test_pred_files = list(model_dir.glob("*_test_preds_*.csv"))
        train_pred_files = list(model_dir.glob("*_train_preds_*.csv"))
        metadata_files = list(model_dir.glob("*_metadata_*.yaml"))
        
        if not test_pred_files:
            print(f"No test prediction files found for {model_name}")
            return None, None, None
        
        # Use latest files
        latest_test = max(test_pred_files, key=lambda x: x.stat().st_mtime)
        latest_train = max(train_pred_files, key=lambda x: x.stat().st_mtime) if train_pred_files else None
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime) if metadata_files else None
        
        print(f"Loading data for {model_name}:")
        print(f"  Test predictions: {latest_test.name}")
        if latest_train:
            print(f"  Train predictions: {latest_train.name}")
        if latest_metadata:
            print(f"  Metadata: {latest_metadata.name}")
        
        # Load data
        test_df = pd.read_csv(latest_test)
        train_df = pd.read_csv(latest_train) if latest_train else None
        
        metadata = None
        if latest_metadata:
            try:
                with open(latest_metadata, 'r') as f:
                    content = f.read()
                    # Filter out problematic NumPy serializations
                    lines = content.split('\n')
                    filtered_lines = []
                    skip_lines = False
                    
                    for line in lines:
                        if 'test_metrics:' in line or 'train_metrics:' in line:
                            skip_lines = True
                            continue
                        if skip_lines and (line.startswith('  ') or line.strip() == ''):
                            continue
                        if skip_lines and not line.startswith(' '):
                            skip_lines = False
                        
                        if not skip_lines:
                            filtered_lines.append(line)
                    
                    filtered_content = '\n'.join(filtered_lines)
                    metadata = yaml.safe_load(filtered_content)
                    
                    # Ensure we have the model name
                    if metadata and 'model' not in metadata:
                        metadata['model'] = model_name
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                metadata = {'model': model_name}
        
        return test_df, train_df, metadata
    
    def calculate_metrics(self, df, set_name="Test"):
        """Calculate and display metrics"""
        if df is None or 'true_count' not in df.columns or 'pred_count' not in df.columns:
            print(f"Cannot calculate metrics for {set_name} set - missing required columns")
            return {}
        
        y_true = df['true_count']
        y_pred = df['pred_count']
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Avoid division by zero for MAPE
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
        print(f"\n{set_name} Set Metrics:")
        print("=" * 20)
        for metric, value in metrics.items():
            print(f"{metric:>6}: {value:10.4f}")
        
        return metrics
    
    def create_visualizations(self, test_df, train_df, model_name, time_range=None):
        """Create and save visualizations with detailed analysis text
        
        Args:
            test_df: Test predictions dataframe
            train_df: Train predictions dataframe (optional)
            model_name: Name of the model
            time_range: Tuple of (start_date, end_date) for time filtering
        """
        # Apply time filtering if specified
        if time_range and 'timestamp' in test_df.columns:
            start_date, end_date = time_range
            test_df = test_df[(test_df['timestamp'] >= start_date) & (test_df['timestamp'] <= end_date)].copy()
            print(f"📅 Filtered to time range: {start_date} to {end_date} ({len(test_df)} samples)")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Model Performance Analysis: {model_name.upper()}', fontsize=16, fontweight='bold')
        
        # Calculate key metrics for analysis text
        y_true = test_df['true_count']
        y_pred = test_df['pred_count']
        residuals = y_true - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # 1. Time Series Plot (Test Set)
        ax1 = axes[0, 0]
        if 'timestamp' in test_df.columns:
            x_data = test_df['timestamp']
        else:
            x_data = range(len(test_df))
        
        ax1.plot(x_data, test_df['true_count'], label='Actual', color='blue', linewidth=1.2, alpha=0.8)
        ax1.plot(x_data, test_df['pred_count'], label='Predicted', color='red', linewidth=1.2, linestyle='--', alpha=0.8)
        ax1.set_title('Time Series: Predictions vs Actual', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time' if 'timestamp' in test_df.columns else 'Index')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add analysis text
        time_text = f"""Analysis: The model shows {'strong' if correlation > 0.8 else 'moderate' if correlation > 0.6 else 'weak'} temporal correlation (r={correlation:.3f}).
        The predictions {'closely follow' if mae < y_true.mean() * 0.1 else 'generally track' if mae < y_true.mean() * 0.2 else 'loosely follow'} the actual patterns.
        {'Seasonal patterns are well captured.' if correlation > 0.7 else 'Some temporal patterns may be missed.'}"""
        ax1.text(0.02, 0.98, time_text, transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7), fontsize=8)
        
        # 2. Scatter Plot
        ax2 = axes[0, 1]
        ax2.scatter(test_df['true_count'], test_df['pred_count'], alpha=0.6, color='blue', s=15)
        
        # Perfect prediction line
        min_val = min(test_df['true_count'].min(), test_df['pred_count'].min())
        max_val = max(test_df['true_count'].max(), test_df['pred_count'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        ax2.set_title('Predicted vs Actual Values', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Actual Count')
        ax2.set_ylabel('Predicted Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add analysis text
        scatter_text = f"""Analysis: Points {'cluster tightly' if rmse < y_true.std() * 0.5 else 'spread moderately' if rmse < y_true.std() else 'spread widely'} around perfect line.
        R²={correlation**2:.3f} indicates {correlation**2*100:.1f}% variance explained.
        {'Excellent prediction accuracy' if correlation > 0.9 else 'Good prediction accuracy' if correlation > 0.8 else 'Moderate prediction accuracy' if correlation > 0.6 else 'Room for improvement'} for this range."""
        ax2.text(0.02, 0.98, scatter_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7), fontsize=8)
        
        # 3. Residuals Distribution
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax3.axvline(0, color='red', linestyle='--', label='Zero Error', linewidth=2)
        ax3.axvline(residuals.mean(), color='orange', linestyle='-', label=f'Mean Error ({residuals.mean():.1f})', linewidth=2)
        ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Residual (Actual - Predicted)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add analysis text
        skewness = residuals.skew()
        bias_text = f"""Analysis: Mean error = {residuals.mean():.2f} ({'systematic bias' if abs(residuals.mean()) > rmse * 0.1 else 'minimal bias'}).
        Distribution is {'highly skewed' if abs(skewness) > 1 else 'moderately skewed' if abs(skewness) > 0.5 else 'approximately normal'} (skew={skewness:.2f}).
        {'Model tends to over-predict' if residuals.mean() < -5 else 'Model tends to under-predict' if residuals.mean() > 5 else 'Model is well-calibrated'}."""
        ax3.text(0.02, 0.98, bias_text, transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7), fontsize=8)
        
        # 4. Error over time
        ax4 = axes[1, 1]
        abs_errors = np.abs(residuals)
        ax4.plot(x_data, abs_errors, color='orange', alpha=0.7, linewidth=1.2)
        ax4.axhline(mae, color='red', linestyle='--', label=f'Mean Absolute Error ({mae:.1f})', linewidth=2)
        ax4.set_title('Absolute Error Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time' if 'timestamp' in test_df.columns else 'Index')
        ax4.set_ylabel('Absolute Error')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add analysis text
        error_stability = abs_errors.std()
        error_text = f"""Analysis: Error consistency - std={error_stability:.2f} ({'very stable' if error_stability < mae * 0.5 else 'stable' if error_stability < mae else 'variable'}).
        Max error = {abs_errors.max():.1f}, occurs {'frequently' if (abs_errors > mae * 2).mean() > 0.1 else 'occasionally'}.
        {'Error patterns suggest model limitations' if abs_errors.max() > mae * 5 else 'Error patterns are reasonable'}."""
        ax4.text(0.02, 0.98, error_text, transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7), fontsize=8)
        
        plt.tight_layout()
        
        # Save plot with time range suffix
        time_suffix = f"_{time_range[0].strftime('%Y%m%d')}_{time_range[1].strftime('%Y%m%d')}" if time_range else ""
        output_file = f"model_analysis_{model_name}{time_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Enhanced visualization saved as: {output_file}")
        
        return fig
    
    def show_data_summary(self, test_df, train_df, model_name):
        """Show data summary and statistics"""
        print(f"\n{'='*50}")
        print(f"DATA SUMMARY FOR {model_name.upper()}")
        print(f"{'='*50}")
        
        print(f"\nTest Set Shape: {test_df.shape}")
        if train_df is not None:
            print(f"Train Set Shape: {train_df.shape}")
        
        print(f"\nTest Set Columns: {list(test_df.columns)}")
        
        if 'timestamp' in test_df.columns:
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            print(f"\nTime Range (Test Set):")
            print(f"  From: {test_df['timestamp'].min()}")
            print(f"  To:   {test_df['timestamp'].max()}")
            print(f"  Duration: {test_df['timestamp'].max() - test_df['timestamp'].min()}")
        
        print(f"\nActual Count Statistics (Test Set):")
        print(test_df['true_count'].describe())
        
        print(f"\nPredicted Count Statistics (Test Set):")
        print(test_df['pred_count'].describe())
        
        # Sample data
        print(f"\nSample Data (First 10 rows):")
        display_cols = ['timestamp', 'true_count', 'pred_count'] if 'timestamp' in test_df.columns else ['true_count', 'pred_count']
        print(test_df[display_cols].head(10))
    
    def get_champion_model_info(self):
        """Get current champion model information"""
        champion_file = Path("models/champion.txt")
        if champion_file.exists():
            with open(champion_file, 'r') as f:
                champion_model = f.read().strip()
            return champion_model
        return "Unknown"
    
    def get_system_architecture_info(self, model_name, metadata):
        """Get system architecture and data encoding information"""
        info = {}
        
        # Champion model status
        champion_model = self.get_champion_model_info()
        info['champion_model'] = champion_model
        info['is_champion'] = model_name == champion_model
        
        # Data encoding status
        if metadata and 'data_type' in metadata:
            data_type = metadata['data_type']
            info['data_type'] = data_type
            if data_type == 'dense':
                info['encoding_status'] = "✅ Using autoencoder embeddings (64-dimensional dense representation)"
            else:
                info['encoding_status'] = "📊 Using scaled wide format (direct feature encoding)"
        else:
            info['encoding_status'] = "❓ Data encoding status unknown"
        
        # MLP-specific architecture information
        if model_name == 'mlp' and metadata and 'best_params' in metadata:
            best_params = metadata['best_params']
            info['mlp_architecture'] = {
                'hidden_layers': best_params.get('hidden_layer_sizes', 'Unknown'),
                'alpha': best_params.get('alpha', 'Unknown'),
                'learning_rate_init': best_params.get('learning_rate_init', 'Unknown'),
                'max_iter': best_params.get('max_iter', 'Unknown')
            }
        
        return info

    def show_system_information(self, model_name, metadata):
        """Display system architecture and model information"""
        print(f"\n{'='*60}")
        print("🏗️  SYSTEM ARCHITECTURE & MODEL INFORMATION")
        print(f"{'='*60}")
        
        # Get system info
        sys_info = self.get_system_architecture_info(model_name, metadata)
        
        # Champion model status
        champion_status = "🏆 CHAMPION MODEL" if sys_info.get('is_champion') else "📈 Non-champion model"
        print(f"\nModel Status: {champion_status}")
        print(f"Current Champion: {sys_info.get('champion_model', 'Unknown')}")
        
        # Data encoding information
        print(f"\nData Architecture:")
        print(f"  {sys_info.get('encoding_status', 'Unknown')}")
        
        if 'data_type' in sys_info:
            if sys_info['data_type'] == 'dense':
                print(f"  • Features preprocessed through autoencoder neural network")
                print(f"  • 64-dimensional compressed representation")
                print(f"  • Better for capturing complex feature interactions")
            else:
                print(f"  • Direct feature scaling and engineering")
                print(f"  • Wide format with original feature structure")
                print(f"  • Better for interpretability and linear relationships")
        
        # MLP-specific information
        if model_name == 'mlp' and 'mlp_architecture' in sys_info:
            arch = sys_info['mlp_architecture']
            print(f"\n🧠 MLP Neural Network Architecture:")
            print(f"  • Hidden Layers: {arch['hidden_layers']} neurons")
            print(f"  • Regularization (alpha): {arch['alpha']:.6f}")
            print(f"  • Learning Rate: {arch['learning_rate_init']:.6f}")
            print(f"  • Max Iterations: {arch['max_iter']}")
            print(f"  • Architecture: Input → {arch['hidden_layers'][0]} → {arch['hidden_layers'][1]} → Output")

    def get_time_range_options(self, test_df):
        """Get available time range options"""
        if 'timestamp' not in test_df.columns:
            return {}
        
        # Convert to datetime if needed
        timestamps = pd.to_datetime(test_df['timestamp'])
        start_date = timestamps.min()
        end_date = timestamps.max()
        
        # Calculate different time range options
        full_duration = end_date - start_date
        
        options = {
            'full': ('Full Range', start_date, end_date),
            'recent_30d': ('Recent 30 Days', max(start_date, end_date - pd.Timedelta(days=30)), end_date),
            'recent_90d': ('Recent 90 Days', max(start_date, end_date - pd.Timedelta(days=90)), end_date),
            'recent_6m': ('Recent 6 Months', max(start_date, end_date - pd.Timedelta(days=180)), end_date),
            'recent_1y': ('Recent 1 Year', max(start_date, end_date - pd.Timedelta(days=365)), end_date),
        }
        
        # Add quarterly options if data spans multiple years
        if full_duration.days > 365:
            current_year = end_date.year
            options['q4'] = (f'Q4 {current_year}', 
                           max(start_date, pd.Timestamp(f'{current_year}-10-01')), end_date)
            options['q3'] = (f'Q3 {current_year}', 
                           max(start_date, pd.Timestamp(f'{current_year}-07-01')), 
                           min(end_date, pd.Timestamp(f'{current_year}-09-30')))
        
        # Filter out options that would be too small
        valid_options = {}
        for key, (label, opt_start, opt_end) in options.items():
            if (opt_end - opt_start).days >= 7:  # At least 7 days
                valid_options[key] = (label, opt_start, opt_end)
        
        return valid_options
    
    def select_time_range(self, test_df):
        """Interactive time range selection"""
        options = self.get_time_range_options(test_df)
        
        if not options:
            print("⚠️ No timestamp data available for time range selection")
            return None
        
        print("📅 Available Time Ranges:")
        for i, (key, (label, start, end)) in enumerate(options.items(), 1):
            duration = end - start
            print(f"  {i}. {label}: {start.date()} to {end.date()} ({duration.days} days)")
        
        choice = input(f"\nSelect time range (1-{len(options)}) or press Enter for full range: ").strip()
        
        if not choice:
            return None
        
        try:
            idx = int(choice) - 1
            keys = list(options.keys())
            if 0 <= idx < len(keys):
                selected_key = keys[idx]
                _, start_date, end_date = options[selected_key]
                return (start_date, end_date)
            else:
                print("❌ Invalid selection, using full range")
                return None
        except ValueError:
            print("❌ Invalid input, using full range")
            return None

    def explore_model(self, model_name, show_plots=True):
        """Complete model exploration with optional time range selection"""
        print(f"\n🔍 EXPLORING MODEL: {model_name.upper()}")
        print("=" * 60)
        
        # Load data
        test_df, train_df, metadata = self.load_model_data(model_name)
        
        if test_df is None:
            print(f"❌ Could not load data for {model_name}")
            return
        
        # Show system architecture information
        self.show_system_information(model_name, metadata)
        
        # Show metadata
        if metadata:
            print(f"\nModel Configuration:")
            for key, value in metadata.items():
                if key != 'best_params':
                    print(f"  {key}: {value}")
            
            if 'best_params' in metadata:
                print(f"\nBest Parameters:")
                for param, value in metadata['best_params'].items():
                    print(f"  {param}: {value}")
        
        # Data summary
        self.show_data_summary(test_df, train_df, model_name)
        
        # Time range selection for visualization
        time_range = None
        if show_plots and 'timestamp' in test_df.columns:
            print(f"\n{'='*50}")
            print("TIME RANGE SELECTION FOR VISUALIZATION")
            print(f"{'='*50}")
            time_range = self.select_time_range(test_df)
        
        # Calculate metrics (always use full dataset for metrics)
        self.calculate_metrics(test_df, "Test")
        if train_df is not None:
            self.calculate_metrics(train_df, "Train")
        
        # Create visualizations (with time range filtering if selected)
        if show_plots:
            self.create_visualizations(test_df, train_df, model_name, time_range)
            plt.show()
        
        print(f"\n✅ Analysis complete for {model_name}")

def main():
    parser = argparse.ArgumentParser(description='Explore model performance and predictions')
    parser.add_argument('--model', help='Specific model to analyze (e.g., random_forest)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--no-plots', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    explorer = ModelExplorer()
    
    if args.list:
        models = explorer.get_available_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")
        return
    
    models = explorer.get_available_models()
    if not models:
        print("❌ No trained models found. Please run model training first.")
        return
    
    if args.model:
        if args.model not in models:
            print(f"❌ Model '{args.model}' not found. Available models: {', '.join(models)}")
            return
        explorer.explore_model(args.model, show_plots=not args.no_plots)
    else:
        print("📊 Available models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        choice = input(f"\nSelect a model (1-{len(models)}) or 'all' for all models: ").strip()
        
        if choice.lower() == 'all':
            for model in models:
                explorer.explore_model(model, show_plots=not args.no_plots)
                print("\n" + "="*60 + "\n")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    explorer.explore_model(models[idx], show_plots=not args.no_plots)
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")

if __name__ == "__main__":
    main()
