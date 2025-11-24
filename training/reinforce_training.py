"""
REINFORCE Training Script - Monte Carlo Policy Gradient
Custom implementation of REINFORCE algorithm with numerical stability improvements
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import MuseumGuideEnv


class PolicyNetwork(nn.Module):
    """Neural network for policy with improved stability"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights with smaller values for stability
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Tanh is more stable than ReLU
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        # Use log_softmax for numerical stability
        return torch.softmax(x, dim=-1)


class REINFORCEAgent:
    """REINFORCE algorithm implementation with stability improvements"""
    
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action using policy network"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Check for NaN in input
        if torch.isnan(state).any():
            print("Warning: NaN detected in state!")
            state = torch.nan_to_num(state, nan=0.0)
        
        probs = self.policy(state)
        
        # Check for NaN or invalid probabilities
        if torch.isnan(probs).any() or (probs < 0).any():
            print(f"Warning: Invalid probs detected: {probs}")
            # Use uniform distribution as fallback
            probs = torch.ones_like(probs) / probs.shape[-1]
        
        # Ensure probabilities sum to 1 (numerical stability)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self):
        """Update policy using REINFORCE with baseline"""
        if len(self.rewards) == 0:
            return 0.0
        
        R = 0
        policy_loss = []
        returns = deque()
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        
        # Convert to tensor
        returns = torch.tensor(list(returns), dtype=torch.float32)
        
        # Check for NaN in returns
        if torch.isnan(returns).any():
            print("Warning: NaN detected in returns!")
            returns = torch.nan_to_num(returns, nan=0.0)
        
        # Normalize returns (baseline) - with stability check
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            # If all returns are the same, don't normalize
            returns = returns - returns.mean()
        
        # Clip returns to prevent extreme values
        returns = torch.clamp(returns, -10, 10)
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            if torch.isnan(log_prob).any():
                print("Warning: NaN in log_prob!")
                continue
            policy_loss.append(-log_prob * R)
        
        if len(policy_loss) == 0:
            return 0.0
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        
        # Check for NaN in loss
        if torch.isnan(policy_loss):
            print("Warning: NaN in policy loss! Skipping update.")
            del self.saved_log_probs[:]
            del self.rewards[:]
            return 0.0
        
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Clear episode memory
        del self.saved_log_probs[:]
        del self.rewards[:]
        
        return policy_loss.item()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def flatten_observation(obs):
    """Flatten dictionary observation to vector with normalization"""
    flat_obs = []
    
    # Agent position (normalize to [0, 1])
    flat_obs.extend(obs['agent_pos'].flatten() / 10.0)
    
    # Visitor engagement (already [0, 1])
    flat_obs.extend(obs['visitor_engagement'].flatten())
    
    # Language preference (one-hot encoding)
    lang_onehot = np.zeros(3)
    lang_onehot[obs['language_pref']] = 1.0
    flat_obs.extend(lang_onehot)
    
    # Time spent (normalize to [0, 1])
    flat_obs.extend(obs['time_spent'].flatten() / 200.0)
    
    # Artifacts viewed (already binary)
    flat_obs.extend(obs['artifacts_viewed'].flatten())
    
    # Interest vector (already normalized)
    flat_obs.extend(obs['interest_vector'].flatten())
    
    # Crowding (already [0, 1])
    flat_obs.extend(obs['crowding'].flatten())
    
    result = np.array(flat_obs, dtype=np.float32)
    
    # Replace any NaN or inf with 0
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    
    return result


def train_reinforce(
    n_episodes=1000,
    learning_rate=5e-4,
    gamma=0.99,
    save_path="models/reinforce/",
    run_name="reinforce_run1",
    eval_freq=100
):
    """
    Train REINFORCE agent on Museum Guide environment
    
    Args:
        n_episodes: Number of training episodes
        learning_rate: Learning rate
        gamma: Discount factor
        save_path: Directory to save models
        run_name: Name for this training run
        eval_freq: Frequency of evaluation
    """
    
    print("=" * 70)
    print("REINFORCE TRAINING - Monte Carlo Policy Gradient")
    print("=" * 70)
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Run Name: {run_name}\n")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    run_path = os.path.join(save_path, run_name)
    os.makedirs(run_path, exist_ok=True)
    
    # Create environment
    env = MuseumGuideEnv()
    
    # Get observation dimensions
    sample_obs, _ = env.reset()
    flat_obs = flatten_observation(sample_obs)
    state_dim = len(flat_obs)
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}\n")
    
    # Create agent
    agent = REINFORCEAgent(state_dim, action_dim, learning_rate, gamma)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    artifacts_viewed_list = []
    engagement_list = []
    best_reward = -float('inf')
    
    print("Starting training...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        # Run episode
        while not done and steps < 300:  # Safety limit
            flat_obs = flatten_observation(obs)
            action = agent.select_action(flat_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            agent.rewards.append(reward)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Update policy
        loss = agent.update_policy()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        losses.append(loss if not np.isnan(loss) else 0.0)
        artifacts_viewed_list.append(info['artifacts_viewed_count'])
        engagement_list.append(info['engagement_level'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_artifacts = np.mean(artifacts_viewed_list[-10:])
            avg_engagement = np.mean(engagement_list[-10:])
            
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Artifacts: {avg_artifacts:.1f} | "
                  f"Engagement: {avg_engagement:.2f} | "
                  f"Loss: {loss:.4f}")
        
        # Evaluate and save best model
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, n_episodes=10)
            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            print(f"Average Evaluation Reward: {eval_reward:.2f}\n")
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(os.path.join(run_path, "best_model.pt"))
                print(f"✓ New best model saved! Reward: {eval_reward:.2f}\n")
    
    # Save final model
    agent.save(os.path.join(run_path, "final_model.pt"))
    print(f"\n✓ Final model saved to: {run_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, losses, 
                        artifacts_viewed_list, engagement_list, run_path)
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_reward = evaluate_agent(agent, env, n_episodes=20)
    print(f"Final Average Reward: {final_reward:.2f}")
    
    env.close()
    
    return agent, run_path


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate agent"""
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 300:
            flat_obs = flatten_observation(obs)
            # Use greedy action during evaluation
            with torch.no_grad():
                state = torch.FloatTensor(flat_obs).unsqueeze(0)
                probs = agent.policy(state)
                action = torch.argmax(probs).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def plot_training_curves(rewards, lengths, losses, artifacts, engagements, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    window = min(50, len(rewards) // 10)
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) > window:
        axes[0, 0].plot(np.convolve(rewards, np.ones(window)/window, mode='valid'), 
                     label=f'Moving Average ({window} episodes)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.3, label='Episode Length')
    if len(lengths) > window:
        axes[0, 1].plot(np.convolve(lengths, np.ones(window)/window, mode='valid'),
                     label=f'Moving Average ({window} episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    axes[1, 0].plot(losses, alpha=0.3, label='Policy Loss')
    if len(losses) > window:
        axes[1, 0].plot(np.convolve(losses, np.ones(window)/window, mode='valid'),
                     label=f'Moving Average ({window} episodes)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Artifacts viewed
    axes[1, 1].plot(artifacts, alpha=0.3, label='Artifacts Viewed')
    if len(artifacts) > window:
        axes[1, 1].plot(np.convolve(artifacts, np.ones(window)/window, mode='valid'),
                     label=f'Moving Average ({window} episodes)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Artifacts')
    axes[1, 1].set_title('Artifacts Viewed per Episode')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Engagement
    axes[2, 0].plot(engagements, alpha=0.3, label='Final Engagement')
    if len(engagements) > window:
        axes[2, 0].plot(np.convolve(engagements, np.ones(window)/window, mode='valid'),
                     label=f'Moving Average ({window} episodes)')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Engagement')
    axes[2, 0].set_title('Final Engagement Level')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Hide unused subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150)
    print(f"✓ Training curves saved to: {os.path.join(save_path, 'training_curves.png')}")
    plt.close()


if __name__ == "__main__":
    # Run multiple configurations (reduced episodes for faster training)
    configs = [
        {"learning_rate": 5e-4, "gamma": 0.99, "n_episodes": 1000, "run_name": "reinforce_run1"},
        {"learning_rate": 1e-3, "gamma": 0.99, "n_episodes": 1000, "run_name": "reinforce_run2"},
        {"learning_rate": 5e-4, "gamma": 0.95, "n_episodes": 1000, "run_name": "reinforce_run3"},
        {"learning_rate": 2e-3, "gamma": 0.99, "n_episodes": 1000, "run_name": "reinforce_run4"},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION {i}/{len(configs)}")
        print(f"{'='*70}\n")
        train_reinforce(**config)