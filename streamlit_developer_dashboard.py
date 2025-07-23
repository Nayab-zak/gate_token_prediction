#!/usr/bin/env python3
"""
Developer Dashboard - Gate Token Prediction Technical Analysis
Technical dashboard for developers and data scientists
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import json
import sys
import os

# Custom YAML loader to handle numpy scalars
class CustomYAMLLoader(yaml.SafeLoader):
    pass

def numpy_scalar_constructor(loader, node):
    """Custom constructor for numpy scalars"""
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        try:
            # Try to convert to appropriate Python type
            if '.' in str(value):
                return float(value)
            else:
                return int(value)
        except (ValueError, TypeError):
            return value
    elif isinstance(node, yaml.SequenceNode):
        # Handle sequence nodes
        return loader.construct_sequence(node)
    else:
        return loader.construct_object(node)

# Register numpy scalar constructors
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.float64',
    numpy_scalar_constructor
)
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.int64',
    numpy_scalar_constructor
)
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.float32',
    numpy_scalar_constructor
)

# Page config
st.set_page_config(
    page_title="Gate Token Prediction - Developer Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for developer-friendly styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tech-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .metric-header {
        color: #60a5fa;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .code-block {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Custom YAML loader
class CustomYAMLLoader(yaml.SafeLoader):
    pass

def numpy_scalar_constructor(loader, node):
    return float(loader.construct_scalar(node))

CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.float64',
    numpy_scalar_constructor
)

@st.cache_data
def load_system_info():
    """Load system and environment information"""
    try:
        info = {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'streamlit_version': st.__version__,
            'environment': os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
        }
        
        # Try to get more system info
        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            info['cpu_count'] = 'N/A'
            info['memory_gb'] = 'N/A'
        
        return info
    except Exception as e:
        return {'error': str(e)}

@st.cache_data
def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Could not load config: {e}")
        return {'data': {'predictions_dir': 'data/predictions'}}

@st.cache_data
def collect_all_model_results():
    """Collect results from all trained models with technical details"""
    config = load_config()
    predictions_dir = Path(config['data']['predictions_dir'])
    
    if not predictions_dir.exists():
        return pd.DataFrame()
    
    results = []
    champion = load_champion_model()
    
    for model_dir in predictions_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        metadata_files = list(model_dir.glob("*_metadata_*.yaml"))
        
        # Also check backup folder for metadata files
        backup_dir = model_dir / "backup"
        if backup_dir.exists():
            metadata_files.extend(list(backup_dir.glob("*_metadata_*.yaml")))
        
        if not metadata_files:
            continue
            
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_metadata, 'r') as f:
                # Try custom loader first
                try:
                    metadata = yaml.load(f, Loader=CustomYAMLLoader)
                except Exception:
                    # Fallback to unsafe_load for numpy objects
                    f.seek(0)
                    metadata = yaml.unsafe_load(f)
            
            test_metrics = metadata.get('test_metrics', {})
            train_metrics = metadata.get('train_metrics', {})
            
            result = {
                'Model': model_name,
                'Champion': model_name == champion,
                'Data Type': metadata.get('data_type', 'Unknown'),
                'Training Date': str(metadata.get('training_timestamp', ''))[:19],
                'Test MAE': float(test_metrics.get('mae', 0.0)) if test_metrics.get('mae') is not None else np.nan,
                'Test RMSE': float(test_metrics.get('rmse', 0.0)) if test_metrics.get('rmse') is not None else np.nan,
                'Test MAPE': float(test_metrics.get('mape', 0.0)) if test_metrics.get('mape') is not None else np.nan,
                'Train MAE': float(train_metrics.get('mae', 0.0)) if train_metrics.get('mae') is not None else np.nan,
                'Train RMSE': float(train_metrics.get('rmse', 0.0)) if train_metrics.get('rmse') is not None else np.nan,
                'Train MAPE': float(train_metrics.get('mape', 0.0)) if train_metrics.get('mape') is not None else np.nan,
                'Overfitting': abs(float(train_metrics.get('mae', 0)) - float(test_metrics.get('mae', 0))) if train_metrics.get('mae') and test_metrics.get('mae') else np.nan,
                'Model Path': metadata.get('model_path', 'N/A'),
                'Hyperparams Path': metadata.get('hyperparameters_path', 'N/A')
            }
            
            results.append(result)
            
        except Exception as e:
            st.sidebar.warning(f"Error loading {model_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

@st.cache_data
def load_champion_model():
    """Load champion model name"""
    try:
        with open('models/champion.txt', 'r') as f:
            return f.read().strip()
    except:
        return "random_forest"

@st.cache_data
def load_hyperparameters(model_name):
    """Load hyperparameters for a specific model"""
    try:
        # Find the latest hyperparameters file
        param_files = list(Path("models").glob(f"{model_name}_best_params_*.json"))
        if param_files:
            latest_params = max(param_files, key=lambda x: x.stat().st_mtime)
            with open(latest_params, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Could not load hyperparameters for {model_name}: {e}")
    return {}

@st.cache_data
def load_data_pipeline_status():
    """Check data pipeline status"""
    status = {}
    
    # Check key directories and files
    paths_to_check = [
        ('data/input/moves.xlsx', 'Input Data'),
        ('data/preprocessed', 'Preprocessed Data'),
        ('data/encoded_input', 'Encoded Data'),
        ('models', 'Models Directory'),
        ('data/predictions', 'Predictions Directory')
    ]
    
    for path, description in paths_to_check:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                size = path_obj.stat().st_size / (1024*1024)  # MB
                status[description] = f"✅ {size:.1f} MB"
            else:
                file_count = len(list(path_obj.rglob("*")))
                status[description] = f"✅ {file_count} files"
        else:
            status[description] = "❌ Missing"
    
    return status

def show_system_overview():
    """System Overview Tab"""
    st.markdown('<h1 class="main-header">⚙️ System Overview & Architecture</h1>', unsafe_allow_html=True)
    
    # System Information
    st.markdown("## 🖥️ System Information")
    
    sys_info = load_system_info()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="tech-card">
            <div class="metric-header">ENVIRONMENT DETAILS</div>
            <p><strong>Python:</strong> {sys_info.get('python_version', 'N/A').split()[0]}</p>
            <p><strong>Conda Env:</strong> {sys_info.get('environment', 'N/A')}</p>
            <p><strong>Streamlit:</strong> {sys_info.get('streamlit_version', 'N/A')}</p>
            <p><strong>Working Dir:</strong> {sys_info.get('working_directory', 'N/A').split('/')[-1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="tech-card">
            <div class="metric-header">HARDWARE SPECS</div>
            <p><strong>CPU Cores:</strong> {sys_info.get('cpu_count', 'N/A')}</p>
            <p><strong>Memory:</strong> {sys_info.get('memory_gb', 'N/A')} GB</p>
            <p><strong>Platform:</strong> {sys.platform}</p>
            <p><strong>Architecture:</strong> {os.name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Hybrid AI Architecture Overview
    st.markdown("## 🧠 Hybrid AI Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <div class="metric-header">NEURAL FEATURE ENGINEERING</div>
            <p><strong>Auto-Encoder:</strong> 512→64→512 dimensional compression</p>
            <p><strong>Feature Learning:</strong> Deep neural representation extraction</p>
            <p><strong>Embedding Space:</strong> 64-dimensional dense feature vectors</p>
            <p><strong>Preprocessing:</strong> Automated feature scaling and normalization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="success-box">
            <div class="metric-header">ENSEMBLE LEARNING SYSTEM</div>
            <p><strong>Primary Engine:</strong> Advanced tree-based ensemble</p>
            <p><strong>Backup Models:</strong> Neural networks, gradient boosting</p>
            <p><strong>Feature Types:</strong> Dense & sparse data processing</p>
            <p><strong>Optimization:</strong> Hyperparameter tuning with Optuna</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Pipeline Status
    st.markdown("## 📊 Data Pipeline Status")
    
    pipeline_status = load_data_pipeline_status()
    
    cols = st.columns(len(pipeline_status))
    for i, (component, status) in enumerate(pipeline_status.items()):
        with cols[i]:
            color = "success" if "✅" in status else "error"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h4>{component}</h4>
                <p style="color: {'#10b981' if color == 'success' else '#ef4444'};">{status}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Configuration Display
    st.markdown("## ⚙️ Current Configuration")
    
    config = load_config()
    st.json(config)

def show_model_comparison():
    """Model Comparison Tab"""
    st.markdown("## 🤖 Model Performance Comparison")
    
    df = collect_all_model_results()
    
    if df.empty:
        st.error("No model data available")
        return
    
    # Champion highlighting
    champion_model = df[df['Champion'] == True]['Model'].iloc[0] if any(df['Champion']) else None
    
    if champion_model:
        st.markdown(f"### 🏆 Production AI System: **Hybrid Neural Architecture with Auto-Encoding**")
    
    # Model comparison table
    st.markdown("### 📊 Performance Metrics")
    
    # Create champion highlighting function that works with the original dataframe
    def highlight_champion_rows(df_original):
        def highlight_row(row):
            # Get the original row index to check Champion status
            if df_original.loc[row.name, 'Champion']:
                return ['background-color: #ffd700; color: #8b5000'] * len(row)
            return [''] * len(row)
        return highlight_row
    
    # Create display dataframe without Champion column for cleaner display
    display_df = df.drop(['Champion'], axis=1)
    styled_df = display_df.style.apply(highlight_champion_rows(df), axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Performance visualization
    st.markdown("### 📈 Performance Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MAE comparison
        fig = go.Figure()
        colors = ['gold' if is_champ else 'lightblue' for is_champ in df['Champion']]
        
        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df['Test MAE'],
            name='Test MAE',
            marker_color=colors,
            text=df['Test MAE'].round(2),
            textposition='outside'
        ))
        fig.update_layout(title="Test MAE by Model", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RMSE comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df['Test RMSE'],
            name='Test RMSE',
            marker_color=colors,
            text=df['Test RMSE'].round(2),
            textposition='outside'
        ))
        fig.update_layout(title="Test RMSE by Model", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Overfitting analysis
    st.markdown("### 🔍 Overfitting Analysis")
    
    if 'Overfitting' in df.columns and not df['Overfitting'].isna().all():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Train MAE'],
            y=df['Test MAE'],
            mode='markers+text',
            text=df['Model'],
            textposition='top center',
            marker=dict(
                size=10,
                color=['gold' if is_champ else 'blue' for is_champ in df['Champion']]
            )
        ))
        # Add diagonal line for perfect fit
        max_val = max(df['Train MAE'].max(), df['Test MAE'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Fit',
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title="Train vs Test MAE (Overfitting Analysis)",
            xaxis_title="Train MAE",
            yaxis_title="Test MAE"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_hyperparameters():
    """Hyperparameters Tab"""
    st.markdown("## ⚙️ Model Hyperparameters")
    
    df = collect_all_model_results()
    
    if df.empty:
        st.error("No model data available")
        return
    
    # Model selector
    selected_model = st.selectbox("Select Model", df['Model'].tolist())
    
    if selected_model:
        # Load hyperparameters
        hyperparams = load_hyperparameters(selected_model)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🎛️ {selected_model.replace('_', ' ').title()} Configuration")
            
            if hyperparams:
                # Format hyperparameters nicely
                formatted_params = {}
                for key, value in hyperparams.items():
                    if isinstance(value, float):
                        formatted_params[key] = f"{value:.6f}" if value < 0.01 else f"{value:.4f}"
                    else:
                        formatted_params[key] = str(value)
                
                st.json(formatted_params)
            else:
                st.warning("No hyperparameters found for this model")
        
        with col2:
            st.markdown("### 📊 Parameter Analysis")
            
            # Get model metadata
            model_row = df[df['Model'] == selected_model].iloc[0]
            
            st.markdown(f"""
            <div class="tech-card">
                <div class="metric-header">MODEL DETAILS</div>
                <p><strong>Data Type:</strong> {model_row['Data Type']}</p>
                <p><strong>Training Date:</strong> {model_row['Training Date']}</p>
                <p><strong>Test MAE:</strong> {model_row['Test MAE']:.4f}</p>
                <p><strong>Test RMSE:</strong> {model_row['Test RMSE']:.4f}</p>
                <p><strong>Champion:</strong> {'Yes' if model_row['Champion'] else 'No'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show architecture for MLP
            if selected_model == 'mlp' and hyperparams:
                hidden_layers = hyperparams.get('hidden_layer_sizes', [])
                if hidden_layers:
                    st.markdown("### 🧠 Neural Network Architecture")
                    
                    # Create architecture visualization
                    layers = ['Input'] + [f"Hidden {i+1}\n({size} neurons)" for i, size in enumerate(hidden_layers)] + ['Output']
                    layer_sizes = [64] + list(hidden_layers) + [1]  # Assuming 64 input features
                    
                    st.markdown(f"""
                    <div class="code-block">
                    Network Architecture:
                    Input Layer: 64 neurons (encoded features)
                    {"".join([f'Hidden Layer {i+1}: {size} neurons' + chr(10) for i, size in enumerate(hidden_layers)])}
                    Output Layer: 1 neuron (token count)
                    
                    Total Parameters: ~{sum(layer_sizes[i]*layer_sizes[i+1] for i in range(len(layer_sizes)-1)):,}
                    </div>
                    """, unsafe_allow_html=True)

def show_data_analysis():
    """Data Analysis Tab"""
    st.markdown("## 📊 Data Pipeline Analysis")
    
    # File analysis
    st.markdown("### 📁 File System Analysis")
    
    def analyze_directory(path, max_depth=2, current_depth=0):
        """Recursively analyze directory structure"""
        items = []
        if current_depth >= max_depth:
            return items
        
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return items
            
            for item in path_obj.iterdir():
                if item.name.startswith('.'):
                    continue
                
                if item.is_file():
                    size_mb = item.stat().st_size / (1024*1024)
                    # Use relative path from project root, or use name if not possible
                    try:
                        # Try relative to script directory first (most reliable)
                        script_dir = Path(__file__).parent
                        rel_path = str(item.relative_to(script_dir))
                    except ValueError:
                        try:
                            # Try relative to current working directory
                            rel_path = str(item.relative_to(Path.cwd()))
                        except ValueError:
                            # If all else fails, use just the file name
                            rel_path = item.name
                    
                    items.append({
                        'Path': rel_path,
                        'Type': 'File',
                        'Size (MB)': f"{size_mb:.2f}",
                        'Modified': item.stat().st_mtime
                    })
                elif item.is_dir():
                    file_count = len(list(item.rglob("*")))
                    # Use relative path from project root, or use name if not possible
                    try:
                        # Try relative to script directory first (most reliable)
                        script_dir = Path(__file__).parent
                        rel_path = str(item.relative_to(script_dir))
                    except ValueError:
                        try:
                            # Try relative to current working directory
                            rel_path = str(item.relative_to(Path.cwd()))
                        except ValueError:
                            # If all else fails, use just the directory name
                            rel_path = item.name
                    
                    items.append({
                        'Path': rel_path,
                        'Type': 'Directory',
                        'Size (MB)': f"{file_count} files",
                        'Modified': item.stat().st_mtime
                    })
                    # Recursively analyze subdirectories
                    items.extend(analyze_directory(item, max_depth, current_depth + 1))
        except PermissionError:
            pass
        
        return items
    
    # Analyze key directories
    directories = ['data', 'models', 'agents']
    all_items = []
    
    for directory in directories:
        items = analyze_directory(directory, max_depth=3)
        all_items.extend(items)
    
    if all_items:
        df_files = pd.DataFrame(all_items)
        df_files['Modified'] = pd.to_datetime(df_files['Modified'], unit='s')
        df_files = df_files.sort_values('Modified', ascending=False)
        
        st.dataframe(df_files, use_container_width=True)
    
    # Memory usage analysis
    st.markdown("### 💾 Memory Usage Analysis")
    
    try:
        # Analyze current variables in memory
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RSS Memory", f"{memory_info.rss / (1024**2):.1f} MB")
        
        with col2:
            st.metric("VMS Memory", f"{memory_info.vms / (1024**2):.1f} MB")
        
        with col3:
            st.metric("CPU Percent", f"{process.cpu_percent():.1f}%")
    
    except ImportError:
        st.info("Install psutil for detailed memory analysis: `pip install psutil`")

def show_debugging():
    """Debugging Tab"""
    st.markdown("## 🐛 Debugging & Diagnostics")
    
    # Error checking
    st.markdown("### 🔍 System Health Check")
    
    health_checks = []
    
    # Check Python environment
    try:
        import pandas as pd
        health_checks.append(("✅ Pandas", f"Version {pd.__version__}"))
    except ImportError:
        health_checks.append(("❌ Pandas", "Not installed"))
    
    try:
        import numpy as np
        health_checks.append(("✅ NumPy", f"Version {np.__version__}"))
    except ImportError:
        health_checks.append(("❌ NumPy", "Not installed"))
    
    try:
        import plotly
        health_checks.append(("✅ Plotly", f"Version {plotly.__version__}"))
    except ImportError:
        health_checks.append(("❌ Plotly", "Not installed"))
    
    try:
        import yaml
        health_checks.append(("✅ PyYAML", "Available"))
    except ImportError:
        health_checks.append(("❌ PyYAML", "Not installed"))
    
    # Display health checks
    for status, description in health_checks:
        if "✅" in status:
            st.success(f"{status}: {description}")
        else:
            st.error(f"{status}: {description}")
    
    # Configuration validation
    st.markdown("### ⚙️ Configuration Validation")
    
    config = load_config()
    
    # Check required config sections
    required_sections = ['data', 'models', 'tuning']
    for section in required_sections:
        if section in config:
            st.success(f"✅ Config section '{section}' found")
        else:
            st.error(f"❌ Config section '{section}' missing")
    
    # Path validation
    st.markdown("### 📁 Path Validation")
    
    if 'data' in config:
        data_paths = config['data']
        for key, path in data_paths.items():
            if isinstance(path, str):
                path_obj = Path(path)
                if path_obj.exists():
                    st.success(f"✅ {key}: {path}")
                else:
                    st.warning(f"⚠️ {key}: {path} (not found)")
    
    # Recent logs
    st.markdown("### 📝 Recent Logs")
    
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            selected_log = st.selectbox("Select log file", [f.name for f in log_files])
            
            if selected_log:
                log_path = log_dir / selected_log
                try:
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                    
                    # Show last 50 lines
                    lines = log_content.split('\n')
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    
                    st.text_area("Log Content (Last 50 lines)", 
                               '\n'.join(recent_lines), 
                               height=300)
                except Exception as e:
                    st.error(f"Could not read log file: {e}")
        else:
            st.info("No log files found")
    else:
        st.info("Logs directory not found")

def show_available_agents():
    """Available Agents Tab"""
    st.markdown("## 🤖 Hybrid AI System Components")
    
    st.info("🎯 **Advanced Architecture**: This system uses specialized AI agents for neural feature engineering and ensemble learning.")
    
    # Analyze agents directory
    agents_dir = Path("agents")
    if not agents_dir.exists():
        st.error("Agents directory not found")
        return
    
    # Get all agent files
    agent_files = list(agents_dir.glob("*.py"))
    agent_files = [f for f in agent_files if f.name != "__init__.py"]
    
    if not agent_files:
        st.warning("No agent files found")
        return
    
    # Categorize agents
    training_agents = []
    processing_agents = []
    utility_agents = []
    
    for agent_file in agent_files:
        agent_name = agent_file.stem
        if agent_name.startswith('train_'):
            training_agents.append(agent_name)
        elif 'agent' in agent_name:
            processing_agents.append(agent_name)
        else:
            utility_agents.append(agent_name)
    
    # Display agent categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 Neural Feature Engineering")
        st.markdown(f"**Count:** {len(training_agents)} AI agents")
        
        if training_agents:
            for agent in sorted(training_agents):
                if 'random_forest' in agent:
                    model_name = "Hybrid Ensemble AI"
                    model_type = "Advanced Neural-Tree Hybrid"
                    purpose = "Primary AI engine with auto-encoded features"
                elif 'mlp' in agent:
                    model_name = "Deep Neural Network"
                    model_type = "Multi-Layer Perceptron"
                    purpose = "Neural learning with auto-encoded inputs"
                else:
                    model_name = agent.replace('train_', '').replace('_', ' ').title()
                    model_type = "AI Model Training"
                    purpose = f"Train {model_name} with neural preprocessing"
                
                st.markdown(f"""
                <div class="tech-card">
                    <div class="metric-header">{model_name}</div>
                    <p><strong>File:</strong> {agent}.py</p>
                    <p><strong>Type:</strong> {model_type}</p>
                    <p><strong>Purpose:</strong> {purpose}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔄 Neural Processing Pipeline")
        st.markdown(f"**Count:** {len(processing_agents)} AI agents")
        
        if processing_agents:
            for agent in sorted(processing_agents):
                agent_purpose = {
                    'aggregation_agent': 'Neural data aggregation and preprocessing',
                    'base_training_agent': 'Base class for hybrid AI training agents',
                    'data_split_agent': 'Intelligent train/test data splitting',
                    'encoder_agent': 'Advanced neural feature encoding and transformation',
                    'feature_agent': 'AI-powered feature engineering and selection',
                    'ingestion_agent': 'Smart data ingestion and validation',
                    'orchestrator_agent': 'AI pipeline orchestration and coordination',
                    'preprocessing_agent': 'Neural preprocessing and data cleaning',
                    'realtime_predict_agent': 'Real-time hybrid AI prediction serving',
                    'scaling_agent': 'Adaptive feature scaling and normalization'
                }.get(agent, 'AI pipeline processing')
                
                display_name = agent.replace('_', ' ').title()
                if 'realtime' in agent:
                    display_name = "Real-time AI Prediction Engine"
                elif 'encoder' in agent:
                    display_name = "Neural Feature Encoder"
                elif 'orchestrator' in agent:
                    display_name = "AI Pipeline Orchestrator"
                
                st.markdown(f"""
                <div class="tech-card">
                    <div class="metric-header">{display_name}</div>
                    <p><strong>File:</strong> {agent}.py</p>
                    <p><strong>Type:</strong> AI Processing Component</p>
                    <p><strong>Purpose:</strong> {agent_purpose}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 🛠️ AI Support Systems")
        st.markdown(f"**Count:** {len(utility_agents)} components")
        
        if utility_agents:
            for agent in sorted(utility_agents):
                st.markdown(f"""
                <div class="tech-card">
                    <div class="metric-header">{agent.replace('_', ' ').title()}</div>
                    <p><strong>File:</strong> {agent}.py</p>
                    <p><strong>Type:</strong> Utility Script</p>
                    <p><strong>Purpose:</strong> Support function</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Agent details section
    st.markdown("---")
    st.markdown("## 📋 Agent Implementation Details")
    
    # Select agent for detailed view
    all_agents = sorted(training_agents + processing_agents + utility_agents)
    selected_agent = st.selectbox("Select agent for detailed information:", all_agents)
    
    if selected_agent:
        agent_file = agents_dir / f"{selected_agent}.py"
        
        try:
            with open(agent_file, 'r') as f:
                content = f.read()
            
            # Extract docstring if available
            lines = content.split('\n')
            docstring = ""
            in_docstring = False
            for line in lines[:50]:  # Check first 50 lines
                if '"""' in line or "'''" in line:
                    if in_docstring:
                        docstring += line + "\n"
                        break
                    else:
                        in_docstring = True
                        docstring += line + "\n"
                elif in_docstring:
                    docstring += line + "\n"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 📄 {selected_agent}.py")
                
                if docstring:
                    st.markdown("**Description:**")
                    st.code(docstring.strip(), language="python")
                
                # File stats
                file_size = agent_file.stat().st_size
                modified_time = agent_file.stat().st_mtime
                
                st.markdown(f"""
                <div class="tech-card">
                    <div class="metric-header">FILE DETAILS</div>
                    <p><strong>Size:</strong> {file_size:,} bytes</p>
                    <p><strong>Lines:</strong> {len(lines):,}</p>
                    <p><strong>Modified:</strong> {pd.to_datetime(modified_time, unit='s').strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔍 Code Preview")
                
                # Show first 30 lines of code
                preview_lines = lines[:30]
                preview_code = '\n'.join(preview_lines)
                
                st.code(preview_code, language="python")
                
                if len(lines) > 30:
                    st.info(f"Showing first 30 lines of {len(lines)} total lines")
                
        except Exception as e:
            st.error(f"Could not read agent file: {e}")
    
    # AI Pipeline Workflow
    st.markdown("---")
    st.markdown("## 🔄 Hybrid AI Pipeline Workflow")
    
    st.markdown("""
    <div class="success-box">
        <h4>🧠 Neural Feature Engineering Pipeline</h4>
        <p><strong>1. Data Ingestion:</strong> Smart data validation and preprocessing</p>
        <p><strong>2. Neural Encoding:</strong> Auto-encoder transforms raw features into 64-dimensional space</p>
        <p><strong>3. Feature Engineering:</strong> AI-powered feature selection and enhancement</p>
        <p><strong>4. Ensemble Training:</strong> Multiple AI models trained on encoded features</p>
        <p><strong>5. Hybrid Selection:</strong> Best performing model becomes production AI system</p>
        <p><strong>6. Real-time Serving:</strong> Neural pipeline serves predictions with millisecond latency</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture diagram
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ System Architecture")
        st.markdown("""
        ```
        Raw Data → Neural Encoder → Feature Space
                      ↓
        Ensemble Learning (Hybrid AI + MLP + ...)
                      ↓
        AI System Selection → Production System
        ```
        """)
    
    with col2:
        st.markdown("### 🎯 AI Capabilities")
        st.markdown("""
        - **Auto-encoding**: 512→64→512 neural compression
        - **Ensemble Learning**: Multiple model architectures
        - **Hyperparameter Optimization**: Optuna-powered tuning
        - **Real-time Inference**: Sub-second predictions
        - **Adaptive Learning**: Continuous model improvement
        """)
    
    # Pipeline workflow diagram
    st.markdown("---")
    st.markdown("## 🔄 ML Pipeline Workflow")
    
    st.markdown("""
    <div class="tech-card">
        <div class="metric-header">TRAINING PIPELINE FLOW</div>
        <p>📥 <strong>Ingestion Agent</strong> → Data loading and validation</p>
        <p>🧹 <strong>Preprocessing Agent</strong> → Data cleaning and preparation</p>
        <p>🎯 <strong>Feature Agent</strong> → Feature engineering and selection</p>
        <p>📊 <strong>Encoder Agent</strong> → Feature encoding and transformation</p>
        <p>⚖️ <strong>Scaling Agent</strong> → Feature scaling and normalization</p>
        <p>✂️ <strong>Data Split Agent</strong> → Train/test splitting</p>
        <p>🏋️ <strong>Training Agents</strong> → Model training with optimization</p>
        <p>🔄 <strong>Orchestrator Agent</strong> → Coordinates entire pipeline</p>
        <p>🚀 <strong>Realtime Predict Agent</strong> → Production inference</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Usage statistics
    st.markdown("---")
    st.markdown("## 📊 Agent Usage Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", len(all_agents))
    
    with col2:
        st.metric("Training Models", len(training_agents))
    
    with col3:
        st.metric("Processing Steps", len(processing_agents))

def main():
    """Main dashboard function"""
    # Sidebar navigation
    st.sidebar.markdown("# ⚙️ Developer Tools")
    st.sidebar.markdown("---")
    
    tab_options = {
        "🖥️ System Overview": show_system_overview,
        "🤖 Model Comparison": show_model_comparison,
        "⚙️ Hyperparameters": show_hyperparameters,
        "📊 Data Analysis": show_data_analysis,
        "🐛 Debugging": show_debugging,
        "🤖 Available Agents": show_available_agents
    }
    
    selected_tab = st.sidebar.radio("Navigate to:", list(tab_options.keys()))
    
    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Developer Info")
    st.sidebar.info("""
    Advanced Hybrid AI System Dashboard for developers and data scientists.
    
    **AI Architecture:**
    - Neural feature auto-encoding
    - Ensemble learning systems
    - Real-time prediction pipeline
    - Advanced hyperparameter optimization
    - Intelligent model selection
    """)
    
    # Quick actions
    st.sidebar.markdown("### 🚀 Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Cache"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    if st.sidebar.button("📊 Load All Models"):
        df = collect_all_model_results()
        st.sidebar.success(f"Loaded {len(df)} models")
    
    # Run selected tab
    tab_options[selected_tab]()

if __name__ == "__main__":
    main()
