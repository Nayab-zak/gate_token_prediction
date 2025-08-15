#!/bin/bash
# Production Deployment Verification Script

set -e

echo "🚀 PRODUCTION DEPLOYMENT VERIFICATION"
echo "=================================================="

# Check file structure
echo "📁 Checking file structure..."
required_files=(
    "manage.sh"
    "health_check.sh"
    "config.py"
    ".env"
    "requirements.txt"
    "README.md"
    "PRODUCTION_DEPLOYMENT.md"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "❌ Missing files: ${missing_files[*]}"
    exit 1
else
    echo "✅ All required files present"
fi

# Check directories
echo "📁 Checking directories..."
required_dirs=("agents" "config" "utils" "data" "logs" "models")
missing_dirs=()
for dir in "${required_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        missing_dirs+=("$dir")
    fi
done

if [[ ${#missing_dirs[@]} -gt 0 ]]; then
    echo "❌ Missing directories: ${missing_dirs[*]}"
    exit 1
else
    echo "✅ All required directories present"
fi

# Check permissions
echo "🔐 Checking file permissions..."
if [[ -x "manage.sh" && -x "health_check.sh" ]]; then
    echo "✅ Script files are executable"
else
    echo "⚠️  Script files may not be executable"
fi

# Check Python
echo "🐍 Checking Python..."
if python3 -c "import sys; print(f'Python {sys.version}')"; then
    echo "✅ Python 3 is available"
else
    echo "❌ Python 3 not found"
    exit 1
fi

# Check if we can import our config
echo "⚙️  Checking configuration..."
if python3 -c "from config import settings; print('✅ Configuration loaded')"; then
    echo "✅ Configuration is valid"
else
    echo "❌ Configuration has issues"
    exit 1
fi

# Final summary
echo "=================================================="
echo "✅ SYSTEM READY FOR PRODUCTION DEPLOYMENT!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Run health check: ./health_check.sh --full"
echo "2. Test training pipeline: ./manage.sh history"  
echo "3. Test prediction pipeline: ./manage.sh realtime"
echo "4. Set up cron jobs for automation"
echo "5. Configure monitoring and alerts"
echo ""
echo "📖 See PRODUCTION_DEPLOYMENT.md for detailed steps"
