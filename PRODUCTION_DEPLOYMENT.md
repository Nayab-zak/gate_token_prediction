# Production Deployment Guide

## 🚀 Production Readiness Checklist

### Prerequisites
- [ ] Python 3.10+ installed
- [ ] Vertica database accessible
- [ ] Minimum 5GB disk space available  
- [ ] Minimum 2GB RAM available
- [ ] Required Python packages installed (`pip install -r requirements.txt`)

### Pre-Deployment Validation

1. **Run Health Check**:
   ```bash
   ./health_check.sh --full
   ```

2. **Validate Production Configuration**:
   ```bash
   python3 config/production.py
   ```

3. **Test Both Pipelines**:
   ```bash
   ./manage.sh history   # Train and validate model
   ./manage.sh realtime  # Test prediction pipeline
   ```

### Environment Configuration

1. **Set Production Environment Variables** in `.env`:
   ```bash
   # Database (CRITICAL - update for production)
   VERTICA_HOST=your-prod-vertica-host
   VERTICA_PORT=5433
   VERTICA_USER=prod_ml_user
   VERTICA_PASSWORD=secure_password
   VERTICA_DB=your_prod_database
   
   # File Management (recommended for production)
   REPLACE_INTERMEDIATE_FILES=true
   KEEP_LAST_N_VERSIONS=3
   
   # Ingestion mode
   INGEST_MODE=realtime  # or history for training
   WINDOW_DAYS=365       # Ensure sufficient data for features
   ```

2. **Security Considerations**:
   - Store sensitive credentials in environment variables or secrets manager
   - Restrict file permissions: `chmod 600 .env`
   - Use dedicated database user with minimal required permissions

### Deployment Steps

1. **Clone to Production Server**:
   ```bash
   git clone <repository> /opt/ml-pipeline
   cd /opt/ml-pipeline
   ```

2. **Set Up Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with production values
   nano .env
   ```

4. **Validate Setup**:
   ```bash
   ./health_check.sh --full
   ```

5. **Initial Training** (if needed):
   ```bash
   ./manage.sh history
   ```

### Production Operations

#### Scheduled Operations (Crontab Examples)

1. **Daily Model Training** (3 AM):
   ```bash
   0 3 * * * cd /opt/ml-pipeline && ./manage.sh history >> logs/cron.log 2>&1
   ```

2. **Hourly Predictions**:
   ```bash
   0 * * * * cd /opt/ml-pipeline && ./manage.sh realtime >> logs/cron.log 2>&1
   ```

3. **Daily Health Checks** (6 AM):
   ```bash
   0 6 * * * cd /opt/ml-pipeline && ./health_check.sh --full --alerts >> logs/health.log 2>&1
   ```

4. **Weekly Cleanup** (Sunday 2 AM):
   ```bash
   0 2 * * 0 cd /opt/ml-pipeline && python3 cleanup_files.py >> logs/cleanup.log 2>&1
   ```

#### Monitoring Commands

```bash
# Check system status
./health_check.sh

# View recent logs
tail -f logs/pipeline.log

# Check last pipeline run
ls -la data/_reports/pipeline/ | tail -5

# Monitor disk usage
df -h

# Check model age
stat models/catboost/model.cbm
```

### Alerting Setup

1. **Email Alerts**:
   ```bash
   export ALERT_EMAIL="admin@yourcompany.com"
   ./health_check.sh --full --alerts
   ```

2. **Slack Integration**:
   ```bash
   export SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
   ./health_check.sh --full --alerts
   ```

### Performance Tuning

1. **Database Connection Pooling**: Consider implementing for high-frequency predictions
2. **Parallel Processing**: Modify agents to use multiprocessing for large datasets
3. **Caching**: Implement feature caching for repeated predictions
4. **Resource Limits**: Set appropriate memory limits using `ulimit`

### Backup Strategy

1. **Model Backup**:
   ```bash
   cp -r models/ /backup/models_$(date +%Y%m%d)/
   ```

2. **Configuration Backup**:
   ```bash
   cp .env config/ /backup/config_$(date +%Y%m%d)/
   ```

### Troubleshooting

#### Common Issues:

1. **Database Connection Timeout**:
   - Check network connectivity
   - Verify credentials and permissions
   - Increase connection timeout in Vertica settings

2. **Insufficient Memory**:
   - Reduce batch size in `.env`: `BATCH_ROWS=50000`
   - Add swap space if needed

3. **Model Performance Degradation**:
   - Retrain model more frequently
   - Check data quality and feature drift
   - Monitor prediction accuracy over time

#### Log Locations:
- Pipeline logs: `logs/pipeline.log`
- Individual agent logs: `logs/<agent_name>.log`  
- Health check logs: `logs/health.log`
- Cron job logs: `logs/cron.log`

### Security Hardening

1. **File Permissions**:
   ```bash
   chmod 600 .env                    # Protect credentials
   chmod 755 *.sh                    # Make scripts executable
   chmod -R 755 logs/                # Log directory access
   ```

2. **Database Security**:
   - Use dedicated ML user with minimal permissions
   - Enable SSL connections if available
   - Regular password rotation

3. **System Security**:
   - Keep system packages updated
   - Use firewall rules to restrict access
   - Regular security audits

## 🎯 Success Metrics

Your production deployment is successful when:
- ✅ Health checks pass consistently
- ✅ Predictions are generated and deployed automatically
- ✅ Model performance remains stable
- ✅ No critical errors in logs for 7+ days
- ✅ Pipeline completion time < 15 minutes
- ✅ Database predictions table updated regularly

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run health check: `./health_check.sh --full`
3. Review this deployment guide
4. Contact ML team with specific error messages and log excerpts
