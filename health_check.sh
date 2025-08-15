#!/bin/bash
# Production Deployment Health Checks and Monitoring
# Usage: ./health_check.sh [--full] [--alerts]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="logs"
ALERT_EMAIL="${ALERT_EMAIL:-admin@company.com}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

print_status() {
    echo -e "${2}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() { print_status "$1" "$GREEN"; }
print_warning() { print_status "$1" "$YELLOW"; }
print_error() { print_status "$1" "$RED"; }

check_system_health() {
    print_status "🏥 Starting system health check..." "$GREEN"
    
    # Check disk space
    disk_usage=$(df -h . | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 85 ]; then
        print_error "❌ Disk usage critical: ${disk_usage}%"
        return 1
    elif [ "$disk_usage" -gt 70 ]; then
        print_warning "⚠️ Disk usage high: ${disk_usage}%"
    else
        print_success "✅ Disk usage OK: ${disk_usage}%"
    fi
    
    # Check memory usage
    memory_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d. -f1)
    if [ "$memory_usage" -gt 90 ]; then
        print_error "❌ Memory usage critical: ${memory_usage}%"
        return 1
    elif [ "$memory_usage" -gt 80 ]; then
        print_warning "⚠️ Memory usage high: ${memory_usage}%"
    else
        print_success "✅ Memory usage OK: ${memory_usage}%"
    fi
    
    # Check log directory
    if [ ! -d "$LOG_DIR" ]; then
        print_error "❌ Log directory missing: $LOG_DIR"
        return 1
    else
        log_count=$(find "$LOG_DIR" -name "*.log" | wc -l)
        print_success "✅ Log directory OK ($log_count log files)"
    fi
    
    # Check model files
    if [ ! -f "models/catboost/model.cbm" ]; then
        print_error "❌ Model file missing: models/catboost/model.cbm"
        return 1
    else
        model_age=$(find models/catboost/model.cbm -mtime +7)
        if [ -n "$model_age" ]; then
            print_warning "⚠️ Model is older than 7 days"
        else
            print_success "✅ Model file OK"
        fi
    fi
    
    return 0
}

check_pipeline_health() {
    print_status "🔄 Checking pipeline health..." "$GREEN"
    
    # Check recent pipeline runs
    recent_reports=$(find data/_reports/pipeline -name "*.json" -mtime -1 2>/dev/null | wc -l)
    if [ "$recent_reports" -eq 0 ]; then
        print_warning "⚠️ No pipeline runs in last 24 hours"
    else
        print_success "✅ Found $recent_reports recent pipeline runs"
    fi
    
    # Check for pipeline errors in logs
    if [ -f "$LOG_DIR/pipeline.log" ]; then
        recent_errors=$(grep -c "ERROR\|FAILED" "$LOG_DIR/pipeline.log" 2>/dev/null || echo 0)
        if [ "$recent_errors" -gt 0 ]; then
            print_error "❌ Found $recent_errors errors in pipeline log"
            # Show last few errors
            echo "Recent errors:"
            grep "ERROR\|FAILED" "$LOG_DIR/pipeline.log" | tail -3
            return 1
        else
            print_success "✅ No recent pipeline errors"
        fi
    fi
    
    return 0
}

check_database_connectivity() {
    print_status "🗄️ Checking database connectivity..." "$GREEN"
    
    # Test database connection using Python
    python3 -c "
from agents.ingestion_agent import _connect
try:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
            result = cur.fetchone()
    print('✅ Database connection successful')
    exit(0)
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
" && print_success "✅ Database connectivity OK" || {
    print_error "❌ Database connection failed"
    return 1
}
}

run_production_validation() {
    print_status "🔍 Running production readiness validation..." "$GREEN"
    
    if python3 config/production.py; then
        print_success "✅ Production validation passed"
        return 0
    else
        print_error "❌ Production validation failed"
        return 1
    fi
}

send_alert() {
    local message="$1"
    local severity="$2"
    
    if [ "$severity" = "critical" ]; then
        # Send email alert if configured
        if command -v mail >/dev/null 2>&1 && [ -n "$ALERT_EMAIL" ]; then
            echo "$message" | mail -s "🚨 ML Pipeline Critical Alert" "$ALERT_EMAIL"
            print_status "📧 Alert sent to $ALERT_EMAIL" "$YELLOW"
        fi
        
        # Send Slack alert if configured
        if [ -n "$SLACK_WEBHOOK" ] && command -v curl >/dev/null 2>&1; then
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"🚨 ML Pipeline Alert: $message\"}" \
                "$SLACK_WEBHOOK" >/dev/null 2>&1 && \
                print_status "💬 Alert sent to Slack" "$YELLOW"
        fi
    fi
}

main() {
    local full_check=false
    local enable_alerts=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                full_check=true
                shift
                ;;
            --alerts)
                enable_alerts=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--full] [--alerts]"
                echo "  --full    Run comprehensive checks including pipeline validation"
                echo "  --alerts  Enable email/Slack alerts for critical issues"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "🚀 ML Pipeline Health Check - $(date)"
    echo "========================================"
    
    local exit_code=0
    local alert_message=""
    
    # Basic system checks (always run)
    if ! check_system_health; then
        exit_code=1
        alert_message="System health check failed"
    fi
    
    # Database connectivity check
    if ! check_database_connectivity; then
        exit_code=1
        alert_message="$alert_message. Database connectivity failed"
    fi
    
    if [ "$full_check" = true ]; then
        # Pipeline health check
        if ! check_pipeline_health; then
            exit_code=1
            alert_message="$alert_message. Pipeline health issues detected"
        fi
        
        # Production readiness validation
        if ! run_production_validation; then
            exit_code=1
            alert_message="$alert_message. Production validation failed"
        fi
    fi
    
    echo "========================================"
    if [ $exit_code -eq 0 ]; then
        print_success "🎉 All health checks passed!"
    else
        print_error "❌ Health check failed!"
        if [ "$enable_alerts" = true ] && [ -n "$alert_message" ]; then
            send_alert "$alert_message" "critical"
        fi
    fi
    
    exit $exit_code
}

# Run main function with all arguments
main "$@"
