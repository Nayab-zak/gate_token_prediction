# Runbook Template

## Data Prep Steps
- Describe how to obtain, clean, and preprocess the raw data.
- List scripts or commands to run for data loading, sanitization, and feature engineering.
- Specify input file locations and expected formats.

## Training Command
- Provide the exact command(s) to train the model, e.g.:
  ```bash
  make train
  # or
  python scripts/train.py
  ```
- Note any required environment variables or configuration files.

## Retrain Cadence
- Specify how often the model should be retrained (e.g., weekly, monthly, on new data arrival).
- List any automation or scheduling (e.g., cron jobs, CI/CD pipeline).

## Troubleshooting
- Common issues and solutions (e.g., missing data, dependency errors, DB connection problems).
- Where to find logs (logs/ directory) and how to interpret them.
- How to rerun failed steps or resume from checkpoints.

## Contact Points
- List primary contacts for support (name, email, Slack, etc.).
- Escalation path for urgent issues.
