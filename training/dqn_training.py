# dqn_training.py
"""
Training script for DQN using Stable Baselines3
Run: python dqn_training.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environment.custom_env import MuseumGuideEnv
import matplotlib.pyplot as plt

# Create directories
os.makedirs("models/dqn", exist_ok=True)

# Hyperparameter combinations (10 runs)
hyperparams = [
    {"learning_rate": 1e-3, "buffer_size": 10000, "learning_starts": 100, "train_freq": 4, "target_update_interval": 1000, "gamma": 0.99, "tau": 1.0},
    {"learning_rate": 5e-4, "buffer_size": 20000, "learning_starts": 200, "train_freq": 8, "target_update_interval": 500, "gamma": 0.95, "tau": 0.5},
    {"learning_rate": 1e-4, "buffer_size": 5000, "learning_starts": 50, "train_freq": 2, "target_update_interval": 2000, "gamma": 0.98, "tau": 0.8},
    {"learning_rate": 2e-4, "buffer_size": 15000,   "learning_starts": 150, "train_freq": 4, "target_update_interval": 1000, "gamma": 0.97, "tau": 0.9},
    {"learning_rate": 3e-4, "buffer_size": 8000, "learning_starts": 80, "train_freq": 6, "target_update_interval": 1500, "gamma": 0.96, "tau": 0.7},
    {"learning_rate": 7e-4, "buffer_size": 12000, "learning_starts": 120, "train_freq": 5, "target_update_interval": 800, "gamma": 0.99, "tau": 1.0},
    {"learning_rate": 4e-4, "buffer_size": 25000, "learning_starts": 250, "train_freq": 10, "target_update_interval": 600, "gamma": 0.95, "tau": 0.6},
    {"learning_rate": 6e-4, "buffer_size": 30000, "learning_starts": 300, "train_freq": 12, "target_update_interval": 400, "gamma": 0.94, "tau": 0.4},
    {"learning_rate": 8e-4, "buffer_size": 6000, "learning_starts": 60, "train_freq": 3, "target_update_interval": 1200, "gamma": 0.98, "tau": 0.85},
    {"learning_rate": 9e-4, "buffer_size": 18000, "learning_starts": 180, "train_freq": 7, "target_update_interval": 700, "gamma": 0.97, "tau": 0.75},]

total_timesteps = 30000  # Adjust based on compute; higher for better training
n_episodes_eval = 10

results = []

for i, params in enumerate(hyperparams):
    print(f"\n--- Training DQN Run {i+1}/10 ---")
    run_dir = f"models/dqn/dqn_run{i+1}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create environment
    env = MuseumGuideEnv()
    
    # Callback for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=5000, n_eval_episodes=n_episodes_eval, deterministic=True, render=False)
    
    # Initialize and train model
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        device="auto",
        **params
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # Save model
    model.save(f"{run_dir}/best_model")
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes_eval)
    results.append({
        "run": i+1,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        **params
    })
    
    print(f"Run {i+1} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()

# Log results to file
np.save("dqn_results.npy", results)
print("\nDQN Training Complete. Results saved to dqn_results.npy")

# Plot results (optional)
runs = [r["run"] for r in results]
mean_rewards = [r["mean_reward"] for r in results]
plt.figure()
plt.plot(runs, mean_rewards, marker='o')
plt.xlabel("Run")
plt.ylabel("Mean Reward")
plt.title("DQN Training Results")
plt.savefig("dqn_results.png")
plt.show()