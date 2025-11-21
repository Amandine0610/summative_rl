"""
PPO Training Script for Museum Guide Agent
Implements Proximal Policy Optimization using Stable Baselines3
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from environment.custom_env import MuseumGuideEnv


def make_env():
    """Create and wrap the environment"""
    env = MuseumGuideEnv(render_mode=None)
    env = Monitor(env)
    return env


def train_ppo(args):
    """
    Train PPO agent with specified hyperparameters
    
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
    # Replace ['numerical_features'] with the actual key names of your Box observation spaces
    print("Observation Space Structure:")
    print(train_env.observation_space.spaces.keys())
    train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    norm_obs_keys=[
        'agent_pos', 
        'crowding', 
        'interest_vector', 
        'time_spent', 
        'visitor_engagement'
    ] 
)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
    eval_env, 
    norm_obs=True, 
    norm_reward=False, 
    training=False,
    norm_obs_keys=[
        'agent_pos', 
        'crowding', 
        'interest_vector', 
        'time_spent', 
        'visitor_engagement'
    ] 
)
    # Define PPO hyperparameters
    model_params = {
        'policy': 'MultiInputPolicy',
        'env': train_env,
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'verbose': 1,
        'tensorboard_log': os.path.join(args.save_path, 'logs'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n" + "="*70)
    print("PPO TRAINING CONFIGURATION")
    print("="*70)
    for param, value in model_params.items():
        if param not in ['policy', 'env']:
            print(f"{param:20s}: {value}")
    print("="*70 + "\n")
    
    # Create PPO model
    model = PPO(**model_params)
    
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
        name_prefix='ppo_model'
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print("Starting training...")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Device: {model.device}\n")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(args.save_path, 'final_model')
    model.save(final_model_path)
    train_env.save(os.path.join(args.save_path, 'vec_normalize.pkl'))
    
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
    
    import json
    with open(os.path.join(args.save_path, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return model, mean_reward, std_reward


def visualize_agent(model_path, n_episodes=5):
    """
    Visualize trained agent
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to visualize
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = MuseumGuideEnv(render_mode='human')
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            # Get action from model (no exploration)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            if done:
                print(f"Episode finished after {steps} steps")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Artifacts viewed: {info['artifacts_viewed_count']}")
                print(f"Final engagement: {info['engagement_level']:.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for Museum Guide')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--save-path', type=str, default='models/ppo',
                       help='Path to save models')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluation frequency')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Checkpoint save frequency')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize trained agent')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for visualization')
    
    args = parser.parse_args()
    
    if args.visualize:
        if args.model_path is None:
            print("Error: --model-path required for visualization")
            return
        visualize_agent(args.model_path)
    else:
        train_ppo(args)


if __name__ == '__main__':
    main()


"""
EXAMPLE USAGE:

1. Basic training:
   python training/ppo_training.py

2. Training with custom hyperparameters:
   python training/ppo_training.py --learning-rate 1e-4 --n-steps 4096 --batch-size 128

3. Extended training:
   python training/ppo_training.py --total-timesteps 500000

4. Visualize trained model:
   python training/ppo_training.py --visualize --model-path models/ppo/best_model.zip

HYPERPARAMETER TUNING RUNS (10+ configurations):

Run 1: Default
python training/ppo_training.py --learning-rate 3e-4 --clip-range 0.2 --ent-coef 0.01

Run 2: Higher learning rate
python training/ppo_training.py --learning-rate 1e-3 --clip-range 0.2 --ent-coef 0.01

Run 3: Lower learning rate
python training/ppo_training.py --learning-rate 1e-4 --clip-range 0.2 --ent-coef 0.01

Run 4: Larger clip range
python training/ppo_training.py --learning-rate 3e-4 --clip-range 0.3 --ent-coef 0.01

Run 5: Smaller clip range
python training/ppo_training.py --learning-rate 3e-4 --clip-range 0.1 --ent-coef 0.01

Run 6: No entropy bonus
python training/ppo_training.py --learning-rate 3e-4 --clip-range 0.2 --ent-coef 0.0

Run 7: Higher entropy
python training/ppo_training.py --learning-rate 3e-4 --clip-range 0.2 --ent-coef 0.05

Run 8: Higher GAE lambda
python training/ppo_training.py --learning-rate 3e-4 --gae-lambda 0.99 --ent-coef 0.01

Run 9: Lower GAE lambda
python training/ppo_training.py --learning-rate 3e-4 --gae-lambda 0.90 --ent-coef 0.01

Run 10: Larger batch size
python training/ppo_training.py --learning-rate 3e-4 --batch-size 128 --ent-coef 0.01

Run 11: Combined best parameters
python training/ppo_training.py --learning-rate 1e-4 --clip-range 0.2 --ent-coef 0.05 --gae-lambda 0.99

Run 12: Aggressive exploration
python training/ppo_training.py --learning-rate 5e-4 --clip-range 0.3 --ent-coef 0.1
"""