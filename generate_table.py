import numpy as np
import pandas as pd
import os

# List of results files (update if needed)
results_files = {
    'DQN': 'dqn_results.npy',
    'PPO': 'ppo_results.npy',
    'A2C': 'a2c_results.npy',
    'REINFORCE': 'reinforce_results.npy'  # Skip if not trained
}

# Column mappings for each algo (adjust based on your hyperparams/results keys)
col_mappings = {
    'DQN': ['run', 'learning_rate', 'gamma', 'buffer_size', 'train_freq', 'mean_reward', 'std_reward'],
    'PPO': ['run', 'learning_rate', 'gamma', 'n_steps', 'clip_range', 'mean_reward', 'std_reward'],
    'A2C': ['run', 'learning_rate', 'gamma', 'n_steps', 'gae_lambda', 'mean_reward', 'std_reward'],
    'REINFORCE': ['run', 'learning_rate', 'gamma', 'mean_reward', 'std_reward']
}

for algo, file_path in results_files.items():
    if not os.path.exists(file_path):
        print(f"Skipping {algo}: {file_path} not found.")
        continue
    
    # Load results
    loaded = np.load(file_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        results = loaded.tolist()  # Convert to list of dicts
    else:
        results = [loaded] if not isinstance(loaded, list) else loaded
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Select columns based on mapping (fallback if missing)
    available_cols = [col for col in col_mappings[algo] if col in df.columns]
    df = df[available_cols]
    
    # Format mean ± std
    if 'mean_reward' in df.columns and 'std_reward' in df.columns:
        df['mean_reward'] = df['mean_reward'].apply(lambda x: f"{x:.1f}")
        df['std_reward'] = df['std_reward'].apply(lambda x: f"± {x:.1f}")
        df['Performance'] = df['mean_reward'] + df['std_reward']  # Combined col for report
        df = df.drop(columns=['mean_reward', 'std_reward'])
        available_cols[-1] = 'Performance'  # Last col is now combined
    
    # Round numerics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    
    # Sort by run
    df = df.sort_values('run').reset_index(drop=True)
    
    # Generate Markdown table header based on algo
    if algo == 'DQN':
        print("### 4.1 DQN\n\n| Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration Strategy | Mean Reward |")
        print("|---------------|-------|--------------------|------------|----------------------|-------------|")
    elif algo == 'REINFORCE':
        print("### 4.2 REINFORCE\n\n| Learning Rate | Gamma | Hidden Size | Baseline Used | Update Freq | Mean Reward |")
        print("|---------------|-------|-------------|---------------|-------------|-------------|")
    elif algo == 'A2C':
        print("### 4.3 A2C\n\n| Learning Rate | Gamma | n_steps | GAE Lambda | Use RMSProp | Mean Reward |")
        print("|---------------|-------|---------|------------|-------------|-------------|")
    elif algo == 'PPO':
        print("### 4.4 PPO\n\n| Learning Rate | Gamma | n_steps | Clip Range | GAE Lambda | Mean Reward |")
        print("|---------------|-------|---------|------------|------------|-------------|")
    
    # Print rows
    for _, row in df.iterrows():
        row_str = " | ".join(str(val) for val in row.values)
        print(f"| {row_str} |")
    
    print("\n")  # Spacer between tables
    
    # Save CSV for easy copy to report
    df.to_csv(f'{algo.lower()}_report_table.csv', index=False)
    print(f"Saved {algo.lower()}_report_table.csv\n")

print("Tables generated! Copy Markdown rows to your report template.")