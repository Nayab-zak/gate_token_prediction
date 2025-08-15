#!/bin/bash
# Production Directory Cleanup Script
# Removes development/testing files and keeps only production essentials

set -e

echo "🧹 Starting production directory cleanup..."

# Files to remove (development/testing only)
FILES_TO_REMOVE=(
    "test_orchestrator.py"
    "test_realtime_orchestrator.py"
    "FILE_MANAGEMENT_SUMMARY.md"
    "PIPELINE_GUIDE.md"
    "REALTIME_ORCHESTRATOR_GUIDE.md"
    ".vscode"
    "catboost_info"
    "requirements.txt"  # Empty file, will create proper one
)

# Directories to remove
DIRS_TO_REMOVE=(
    "__pycache__"
    "agents/__pycache__"
    "utils/__pycache__"
)

# Remove development files
echo "📂 Removing development files..."
for file in "${FILES_TO_REMOVE[@]}"; do
    if [[ -e "$file" ]]; then
        echo "  ❌ Removing: $file"
        rm -rf "$file"
    else
        echo "  ⏩ Not found: $file"
    fi
done

# Remove Python cache directories
echo "🗂️ Removing Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Create proper requirements.txt
echo "📄 Creating production requirements.txt..."
cat > requirements.txt << 'EOF'
# Production Requirements for ML Pipeline
pandas>=1.5.0
numpy>=1.21.0
catboost>=1.2.0
vertica-python>=1.3.0
psutil>=5.9.0
python-dotenv>=0.19.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
EOF

# Create production-specific .gitignore
echo "🔒 Creating production .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# ML Pipeline specific
data/input_raw/
data/preprocessed/
data/features/*/
!data/features/.gitkeep
logs/*.log
models/catboost/predictions_*
data/_reports/
catboost_info/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOF

# Create data directory structure with .gitkeep files
echo "📁 Setting up production data directories..."
mkdir -p data/{input_raw,preprocessed,features}/{history,realtime}
mkdir -p data/_reports/{pipeline,eval}
mkdir -p logs
mkdir -p models/catboost

# Add .gitkeep files to preserve empty directories
touch data/input_raw/.gitkeep
touch data/preprocessed/.gitkeep  
touch data/features/.gitkeep
touch data/_reports/.gitkeep
touch logs/.gitkeep
touch models/.gitkeep

# Set proper permissions
echo "🔐 Setting file permissions..."
chmod 755 manage.sh health_check.sh
chmod 600 .env 2>/dev/null || echo "  ⚠️ .env file not found - create it from .env.example"
chmod 644 *.py
chmod 644 config/*.py
chmod 644 agents/*.py
chmod 644 utils/*.py

# Final structure check
echo "📋 Final directory structure:"
echo "=================="
tree -I '__pycache__|*.pyc|catboost_info' -L 2 2>/dev/null || ls -la

echo ""
echo "✅ Production cleanup complete!"
echo ""
echo "📦 PRODUCTION READY STRUCTURE:"
echo "├── Core Scripts:"
echo "│   ├── manage.sh              # Main interface"
echo "│   ├── health_check.sh        # System monitoring"
echo "│   └── cleanup_files.py       # File maintenance"
echo "├── Configuration:"
echo "│   ├── .env                   # Environment variables"
echo "│   ├── config.py              # Settings"
echo "│   └── config/production.py   # Production validation"
echo "├── Pipeline:"
echo "│   ├── pipeline_orchestrator.py    # Training pipeline"
echo "│   ├── realtime_orchestrator.py    # Prediction pipeline" 
echo "│   └── agents/                     # ML processing agents"
echo "├── Documentation:"
echo "│   ├── README.md              # Overview and usage"
echo "│   └── PRODUCTION_DEPLOYMENT.md   # Deployment guide"
echo "└── Runtime:"
echo "    ├── data/                  # Processing data"
echo "    ├── logs/                  # Application logs"
echo "    ├── models/                # Trained models"
echo "    └── utils/                 # Helper utilities"
echo ""
echo "🚀 Ready for production deployment!"
