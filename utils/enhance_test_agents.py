#!/usr/bin/env python3
"""
Utility to enhance all test agents with transaction-level identifiers.
This script updates all test agents to include TerminalID, MoveType, and Desig
in the output for better transaction tracking.
"""

import os
import re

def update_test_agent(agent_path, model_type):
    """Update a single test agent to include transaction keys"""
    
    print(f"Updating {agent_path}...")
    
    with open(agent_path, 'r') as f:
        content = f.read()
    
    # 1. Update load_test_data function
    old_load_pattern = r'def load_test_data\(\):\s*\n(.*?)return [^,\n]*'
    
    if 'classic' in agent_path:
        input_type = 'classic'
    else:
        input_type = 'augmented'
    
    new_load_function = f'''def load_test_data():
    # Load encoded data for model prediction
    encoded_path = os.path.join(DATA_DIR, 'encoded_input', 'test_input_{input_type}.csv')
    df_encoded = pd.read_csv(encoded_path, parse_dates=['datetime'])
    X = df_encoded.drop(columns=['datetime', 'TokenCount'])
    y = df_encoded['TokenCount']
    timestamps = df_encoded['datetime']
    
    # Load original features to get transaction-level identifiers
    features_path = os.path.join(DATA_DIR, 'features', 'test_features.csv')
    df_features = pd.read_csv(features_path, parse_dates=['datetime'])
    
    # Extract transaction identifiers (ensure same order as encoded data)
    df_features = df_features.sort_values('datetime').reset_index(drop=True)
    transaction_keys = df_features[['MoveType', 'TerminalID', 'Desig']].copy()
    
    return X, y, timestamps, transaction_keys'''
    
    # Replace the load_test_data function
    content = re.sub(
        r'def load_test_data\(\):.*?return [^,\n]*[,\s\w]*', 
        new_load_function,
        content,
        flags=re.DOTALL
    )
    
    # 2. Update evaluate_and_save function signature and body
    if 'lstm' in agent_path or 'mlp' in agent_path:
        # These models use scaler
        old_eval_signature = r'def evaluate_and_save\((.*?), X, y, timestamps, logger\):'
        new_eval_signature = r'def evaluate_and_save(\1, X, y, timestamps, transaction_keys, logger):'
    else:
        # Tree-based models don't use scaler
        old_eval_signature = r'def evaluate_and_save\(model, X, y, timestamps, logger\):'
        new_eval_signature = 'def evaluate_and_save(model, X, y, timestamps, transaction_keys, logger):'
    
    content = re.sub(old_eval_signature, new_eval_signature, content)
    
    # 3. Update the results DataFrame creation
    old_results_pattern = r"results = pd\.DataFrame\(\{'datetime': timestamps, 'actual': y, 'prediction': preds\}\)"
    new_results_pattern = '''results = pd.DataFrame({
        'datetime': timestamps,
        'TerminalID': transaction_keys['TerminalID'],
        'MoveType': transaction_keys['MoveType'], 
        'Desig': transaction_keys['Desig'],
        'actual': y,
        'prediction': preds
    })'''
    
    content = re.sub(old_results_pattern, new_results_pattern, content)
    
    # 4. Update main function call
    if 'lstm' in agent_path or 'mlp' in agent_path:
        # These load scaler
        old_main_pattern = r'X, y, timestamps = load_test_data\(\)\s*\n\s*evaluate_and_save\((.*?), X, y, timestamps, logger\)'
        new_main_pattern = r'X, y, timestamps, transaction_keys = load_test_data()\n    evaluate_and_save(\1, X, y, timestamps, transaction_keys, logger)'
    else:
        # Tree-based models
        old_main_pattern = r'X, y, timestamps = load_test_data\(\)\s*\n\s*evaluate_and_save\(model, X, y, timestamps, logger\)'
        new_main_pattern = 'X, y, timestamps, transaction_keys = load_test_data()\n    evaluate_and_save(model, X, y, timestamps, transaction_keys, logger)'
    
    content = re.sub(old_main_pattern, new_main_pattern, content)
    
    # 5. Add logging about transaction keys
    log_pattern = r'(logger\.info\(f"Saved .* test results to \{results_path\}"\))'
    new_log = r'\1\n    logger.info(f"Results include transaction keys: TerminalID, MoveType, Desig")'
    content = re.sub(log_pattern, new_log, content)
    
    # Write updated content
    with open(agent_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {os.path.basename(agent_path)}")


def main():
    """Update all test agents"""
    
    print("🔄 Enhancing test agents with transaction-level identifiers...")
    
    test_agents_dir = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents/07_test_agents"
    
    # Get all test agent files except the one we already updated
    agent_files = [f for f in os.listdir(test_agents_dir) 
                   if f.endswith('_agent.py') and f != '07_test_catboost_classic_agent.py']
    
    models_updated = []
    
    for agent_file in sorted(agent_files):
        agent_path = os.path.join(test_agents_dir, agent_file)
        
        # Extract model type from filename
        model_type = agent_file.replace('07_test_', '').replace('_agent.py', '')
        
        try:
            update_test_agent(agent_path, model_type)
            models_updated.append(model_type)
        except Exception as e:
            print(f"❌ Error updating {agent_file}: {e}")
    
    print(f"\n🎉 Successfully updated {len(models_updated)} test agents:")
    for model in models_updated:
        print(f"   ✅ {model}")
    
    print(f"\n📋 Updated agents will now include transaction keys in output:")
    print("   • TerminalID (T1, T2, T3, T4)")
    print("   • MoveType (In, Out)")  
    print("   • Desig (EXP, FCL, MT, T/S)")
    print("\n🔍 This will make it easier to understand what each prediction represents!")


if __name__ == '__main__':
    main()
