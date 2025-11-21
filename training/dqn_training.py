"""
DQN Training Script for Museum Guide Agent
Implements Deep Q-Network using Stable Baselines3
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import json
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from environment.custom_env import MuseumGuideEnv


class FlattenDictWrapper(gym.Wrapper):
    """Wrapper to flatten Dict observation space for DQN"""
    def __init__(self, env):
        super().__init__(env)
        # Note: This is a simplified version
        # For production, properly flatten all Dict components
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
    
    def _flatten_obs(self, obs):
        """Flatten dict observation to single array"""
        # Concatenate all observation components
        flat_obs = np.concatenate([
            obs['agent_pos'].flatten(),
            obs['visitor_engagement'].flatten(),
            np.array([obs['language_pref']]),
            obs['time_spent'].flatten(),
            obs['artifacts_viewed'].flatten(),
            obs['interest_vector'].flatten(),
            obs['crowding'].flatten()
        ])
        return flat_obs


def make_env():
    """Create and wrap environment for DQN"""
    env = MuseumGuideEnv(render_mode=None)
    
    # Note: DQN requires flattened observations
    # You may need to modify MuseumGuideEnv or use a wrapper
    # For this example, we'll use the Dict space directly
    # Stable Baselines3 DQN supports dict observations
    
    env = Monitor(env)
    return env


def train_dqn(args):
    """
    Train DQN agent with specified hyperparameters
    
    Args:
        args: Command line arguments containing hyperparameters
    """
    
    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'checkpoints'), exist_ok=True)
    
    # Create training environment
    print("Creating environment...")
    train_env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    # Define DQN hyperparameters
    model_params = {
        'policy': 'MultiInputPolicy',  # Use MultiInputPolicy for Dict observations
        'env': train_env,
        'learning_rate': args.learning_rate,
        'buffer_size': args.buffer_size,
        'learning_starts': args.learning_starts,
        'batch_size': args.batch_size,
        'tau': args.tau,
        'gamma': args.gamma,
        'train_freq': args.train_freq,
        'gradient_steps': args.gradient_steps,
        'target_update_interval': args.target_update_interval,
        'exploration_fraction': args.exploration_fraction,
        'exploration_initial_eps': args.exploration_initial_eps,
        'exploration_final_eps': args.exploration_final_eps,
        'verbose': 1,
        'tensorboard_log': os.path.join(args.save_path, 'logs'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n" + "="*70)
    print("DQN TRAINING CONFIGURATION")
    print("="*70)
    for param, value in model_params.items():
        if param not in ['policy', 'env']:
            print(f"{param:25s}: {value}")
    print("="*70 + "\n")
    
    # Create DQN model
    model = DQN(**model_params)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_path,
        log_path=os.path.join(args.save_path, 'logs'),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.save_path, 'checkpoints'),
        name_prefix='dqn_model'
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print("Starting training...")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Device: {model.device}")
    print(f"Replay buffer size: {args.buffer_size:,}")
    print(f"Exploration: {args.exploration_initial_eps} → {args.exploration_final_eps}")
    print()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(args.save_path, 'final_model')
    model.save(final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")
    
    # Evaluate final model
    print("\nEvaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save training results
    results = {
        'hyperparameters': vars(args),
        'final_mean_reward': float(mean_reward),
        'final_std_reward': float(std_reward),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(args.save_path, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return model, mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for Museum Guide')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='Total training timesteps')
    parser.add_argument('--save-path', type=str, default='models/dqn',
                       help='Path to save models')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluation frequency')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Checkpoint save frequency')
    
    # DQN hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=50000,
                       help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=1000,
                       help='Steps before learning starts')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Minibatch size')
    parser.add_argument('--tau', type=float, default=1.0,
                       help='Soft update coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--train-freq', type=int, default=4,
                       help='Training frequency')
    parser.add_argument('--gradient-steps', type=int, default=1,
                       help='Gradient steps per training')
    parser.add_argument('--target-update-interval', type=int, default=1000,
                       help='Target network update interval')
    parser.add_argument('--exploration-fraction', type=float, default=0.1,
                       help='Fraction of training for exploration')
    parser.add_argument('--exploration-initial-eps', type=float, default=1.0,
                       help='Initial epsilon for exploration')
    parser.add_argument('--exploration-final-eps', type=float, default=0.05,
                       help='Final epsilon for exploration')
    
    args = parser.parse_args()
    
    train_dqn(args)


if __name__ == '__main__':
    main()


"""
EXAMPLE USAGE:

1. Basic training:
   python training/dqn_training.py

2. Training with custom hyperparameters:
   python training/dqn_training.py --learning-rate 5e-4 --buffer-size 100000

3. Extended training with larger buffer:
   python training/dqn_training.py --total-timesteps 300000 --buffer-size 100000

HYPERPARAMETER TUNING RUNS (10+ configurations):

Run 1: Default
python training/dqn_training.py --learning-rate 1e-4 --buffer-size 50000 --batch-size 32

Run 2: Higher learning rate
python training/dqn_training.py --learning-rate 5e-4 --buffer-size 50000 --batch-size 32

Run 3: Lower learning rate
python training/dqn_training.py --learning-rate 5e-5 --buffer-size 50000 --batch-size 32

Run 4: Larger buffer
python training/dqn_training.py --learning-rate 1e-4 --buffer-size 100000 --batch-size 32

Run 5: Larger batch
python training/dqn_training.py --learning-rate 1e-4 --buffer-size 50000 --batch-size 64

Run 6: More frequent updates
python training/dqn_training.py --learning-rate 1e-4 --train-freq 1 --gradient-steps 4

Run 7: Conservative exploration
python training/dqn_training.py --exploration-final-eps 0.1 --exploration-fraction 0.2

Run 8: Aggressive exploration
python training/dqn_training.py --exploration-final-eps 0.01 --exploration-fraction 0.05

Run 9: Faster target updates
python training/dqn_training.py --target-update-interval 500 --tau 0.005

Run 10: Slower target updates
python training/dqn_training.py --target-update-interval 2000 --tau 1.0

Run 11: Combined best (larger buffer + moderate LR)
python training/dqn_training.py --learning-rate 3e-4 --buffer-size 100000 --batch-size 64

Run 12: High capacity (large buffer + batch)
python training/dqn_training.py --learning-rate 1e-4 --buffer-size 150000 --batch-size 128
"""