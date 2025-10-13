import yaml
from app.db.vertica_client import VerticaClient
from app.utils.env import load_config_env

cfg = load_config_env('config/config.yaml')
vc = VerticaClient(load_config_env('config/db.yaml'))

# Check recent predictions
df = vc.fetch_df('dev', '''
    SELECT MoveDate, MoveHour, MoveType, TerminalID, Desig, 
           prediction_ts_utc, model_version, BI_BATCH_ID
    FROM DPW_DL.T_DA_PRED_GATE_TOKEN 
    ORDER BY prediction_ts_utc DESC 
    LIMIT 30
''', {})

print("Recent predictions (showing potential duplicates):")
print(df)

# Check for duplicates in the same business key
print("\n" + "="*50)
print("Checking for duplicate business keys:")

df_counts = vc.fetch_df('dev', '''
    SELECT MoveDate, MoveHour, MoveType, TerminalID, Desig, 
           COUNT(*) as count,
           MIN(prediction_ts_utc) as first_prediction,
           MAX(prediction_ts_utc) as last_prediction
    FROM DPW_DL.T_DA_PRED_GATE_TOKEN 
    GROUP BY MoveDate, MoveHour, MoveType, TerminalID, Desig
    HAVING COUNT(*) > 1
    ORDER BY count DESC, MoveDate DESC
    LIMIT 10
''', {})

print(df_counts)
