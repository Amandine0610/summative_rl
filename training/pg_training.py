"""
Training script for Policy Gradient Methods using Stable Baselines3 (PPO, A2C) and custom REINFORCE
Run: python pg_training.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from environment.custom_env import MuseumGuideEnv
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd  # For optional table export

# Create directories
os.makedirs("models/pg", exist_ok=True)

total_timesteps = 10000  # Your 10k setting
n_episodes_eval = 10

# ---------------- PPO Training ----------------
print("--- Training PPO ---")
os.makedirs("models/pg/ppo", exist_ok=True)

ppo_hyperparams = [
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2},
    {"learning_rate": 1e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2},
    {"learning_rate": 1e-3, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2},
    {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 32, "n_epochs": 5, "gamma": 0.95, "gae_lambda": 0.9, "clip_range": 0.1},
    {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 32, "n_epochs": 5, "gamma": 0.95, "gae_lambda": 0.9, "clip_range": 0.1},
    {"learning_rate": 1e-3, "n_steps": 1024, "batch_size": 32, "n_epochs": 5, "gamma": 0.95, "gae_lambda": 0.9, "clip_range": 0.1},
    {"learning_rate": 3e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 20, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.3},
    {"learning_rate": 1e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 20, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.3},
    {"learning_rate": 1e-3, "n_steps": 4096, "batch_size": 128, "n_epochs": 20, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.3},
    {"learning_rate": 5e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.97, "gae_lambda": 0.92, "clip_range": 0.2},
]

ppo_results = []
for i, params in enumerate(ppo_hyperparams):
    print(f"\n--- PPO Run {i+1}/10 ---")
    run_dir = f"models/pg/ppo/ppo_run{i+1}"
    os.makedirs(run_dir, exist_ok=True)
    
    env = MuseumGuideEnv()
    eval_callback = EvalCallback(env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=5000, n_eval_episodes=n_episodes_eval, deterministic=True, render=False)
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device="auto",
        **params
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{run_dir}/best_model")
    
    # Evaluate (FIX: n_eval_episodes)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes_eval)
    
    # NEW: Track avg artifacts
    artifacts = []
    for _ in range(n_episodes_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        artifacts.append(info['artifacts_viewed_count'])
    avg_artifacts = np.mean(artifacts)
    
    ppo_results.append({"run": i+1, "mean_reward": mean_reward, "std_reward": std_reward, "avg_artifacts": avg_artifacts, **params})
    print(f"PPO Run {i+1} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} | Avg Artifacts: {avg_artifacts:.1f}")
    
    env.close()

np.save("ppo_results.npy", ppo_results)

# ---------------- A2C Training ----------------
print("\n--- Training A2C ---")
os.makedirs("models/pg/a2c", exist_ok=True)

a2c_hyperparams = [
    {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "use_rms_prop": True, "use_sde": False},
    {"learning_rate": 3e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "use_rms_prop": True, "use_sde": False},
    {"learning_rate": 1e-3, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "use_rms_prop": True, "use_sde": False},
    {"learning_rate": 7e-4, "n_steps": 10, "gamma": 0.95, "gae_lambda": 0.95, "use_rms_prop": False, "use_sde": False},  # Fixed: False
    {"learning_rate": 3e-4, "n_steps": 10, "gamma": 0.95, "gae_lambda": 0.95, "use_rms_prop": False, "use_sde": False},  # Fixed: False
    {"learning_rate": 1e-3, "n_steps": 10, "gamma": 0.95, "gae_lambda": 0.95, "use_rms_prop": False, "use_sde": False},  # Fixed: False
    {"learning_rate": 7e-4, "n_steps": 1, "gamma": 0.99, "gae_lambda": 1.0, "use_rms_prop": True, "use_sde": False},
    {"learning_rate": 3e-4, "n_steps": 1, "gamma": 0.99, "gae_lambda": 1.0, "use_rms_prop": True, "use_sde": False},
    {"learning_rate": 1e-3, "n_steps": 1, "gamma": 0.99, "gae_lambda": 1.0, "use_rms_prop": True, "use_sde": False},
    {"learning_rate": 5e-4, "n_steps": 5, "gamma": 0.97, "gae_lambda": 0.98, "use_rms_prop": True, "use_sde": False},
]

a2c_results = []
for i, params in enumerate(a2c_hyperparams):
    print(f"\n--- A2C Run {i+1}/10 ---")
    run_dir = f"models/pg/a2c/a2c_run{i+1}"
    os.makedirs(run_dir, exist_ok=True)
    
    env = MuseumGuideEnv()
    eval_callback = EvalCallback(env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=5000, n_eval_episodes=n_episodes_eval, deterministic=True, render=False)
    
    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=1,
        device="auto",
        **params
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{run_dir}/best_model")
    
    # Evaluate (FIX: n_eval_episodes)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes_eval)
    
    # NEW: Track avg artifacts (same as PPO)
    artifacts = []
    for _ in range(n_episodes_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        artifacts.append(info['artifacts_viewed_count'])
    avg_artifacts = np.mean(artifacts)
    
    a2c_results.append({"run": i+1, "mean_reward": mean_reward, "std_reward": std_reward, "avg_artifacts": avg_artifacts, **params})
    print(f"A2C Run {i+1} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} | Avg Artifacts: {avg_artifacts:.1f}")
    
    env.close()

np.save("a2c_results.npy", a2c_results)

# ---------------- REINFORCE Implementation ----------------
class SimpleREINFORCE(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64):
        super().__init__()
        # Simplified flatten for dict obs (agent_pos + engagement + lang + time + interest; ignore artifacts/crowding for simplicity)
        obs_dim = 2 + 1 + 1 + 1 + 3  # pos(2), eng(1), lang(1), time(1), interest(3)
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.n),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Flatten key obs parts
        flat_x = np.concatenate([x['agent_pos'], x['visitor_engagement'].flatten(), [x['language_pref']], x['time_spent'].flatten(), x['interest_vector']])
        return self.network(torch.tensor(flat_x, dtype=torch.float32))

def train_reinforce(hyperparams_list):
    reinforce_results = []
    for i, params in enumerate(hyperparams_list):
        print(f"\n--- REINFORCE Run {i+1}/10 ---")
        run_dir = f"models/pg/reinforce/reinforce_run{i+1}"
        os.makedirs(run_dir, exist_ok=True)
        
        env = MuseumGuideEnv()
        policy = SimpleREINFORCE(env.observation_space, env.action_space)
        optimizer = optim.Adam(policy.parameters(), lr=params["learning_rate"])
        
        # Training loop (approx timesteps via episodes)
        episode_rewards = deque(maxlen=n_episodes_eval * 2)  # Buffer for eval
        n_episodes_reinforce = 50  # Adjusted for 10k equiv
        for episode in range(n_episodes_reinforce):
            obs, _ = env.reset()
            log_probs = []
            rewards = []
            done = False
            while not done:
                flat_obs = np.concatenate([obs['agent_pos'], obs['visitor_engagement'].flatten(), [obs['language_pref']], obs['time_spent'].flatten(), obs['interest_vector']])
                probs = policy(torch.tensor(flat_obs, dtype=torch.float32))
                action = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[action])
                log_probs.append(log_prob)
                
                next_obs, reward, terminated, _, _ = env.step(action)
                rewards.append(reward)
                obs = next_obs
                done = terminated
            
            # REINFORCE update
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + params["gamma"] * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.stack(policy_loss).sum()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            episode_rewards.append(sum(rewards))
        
        # Evaluate
        mean_reward = np.mean(list(episode_rewards)[-n_episodes_eval:])
        std_reward = np.std(list(episode_rewards)[-n_episodes_eval:])
        
        # Track avg artifacts (manual eval)
        artifacts = []
        policy.eval()  # Eval mode
        for _ in range(n_episodes_eval):
            obs, _ = env.reset()
            done = False
            while not done:
                flat_obs = np.concatenate([obs['agent_pos'], obs['visitor_engagement'].flatten(), [obs['language_pref']], obs['time_spent'].flatten(), obs['interest_vector']])
                with torch.no_grad():
                    probs = policy(torch.tensor(flat_obs, dtype=torch.float32))
                    action = torch.argmax(probs).item()  # Greedy for eval
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            artifacts.append(info['artifacts_viewed_count'])
        avg_artifacts = np.mean(artifacts)
        
        reinforce_results.append({"run": i+1, "mean_reward": mean_reward, "std_reward": std_reward, "avg_artifacts": avg_artifacts, **params})
        print(f"REINFORCE Run {i+1} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} | Avg Artifacts: {avg_artifacts:.1f}")
        
        torch.save(policy.state_dict(), f"{run_dir}/best_model.pth")
        env.close()
    
    np.save("reinforce_results.npy", reinforce_results)
    return reinforce_results

# REINFORCE hyperparams
reinforce_hyperparams = [
    {"learning_rate": 1e-3, "gamma": 0.99},
    {"learning_rate": 5e-4, "gamma": 0.99},
    {"learning_rate": 1e-4, "gamma": 0.99},
    {"learning_rate": 1e-3, "gamma": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.95},
    {"learning_rate": 1e-4, "gamma": 0.95},
    {"learning_rate": 1e-3, "gamma": 0.99},
    {"learning_rate": 5e-4, "gamma": 0.99},
    {"learning_rate": 1e-4, "gamma": 0.99},
    {"learning_rate": 2e-4, "gamma": 0.97},
]

train_reinforce(reinforce_hyperparams)

print("\nPolicy Gradient Training Complete. Results saved as *_results.npy")

# Plot results (optional)
for algo, res in [("PPO", ppo_results), ("A2C", a2c_results)]:
    runs = [r["run"] for r in res]
    mean_rewards = [r["mean_reward"] for r in res]
    plt.figure()
    plt.plot(runs, mean_rewards, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Mean Reward")
    plt.title(f"{algo} Training Results (10k Timesteps)")
    plt.savefig(f"{algo.lower()}_results.png")
    plt.show()