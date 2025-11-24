"""
A2C Training Script - Actor-Critic RL
"""

import os
import sys
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import MuseumGuideEnv


def train_a2c(
    total_timesteps=200000,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_path="models/a2c/",
    run_name="a2c_run1"
):
    """
    Train A2C agent on Museum Guide environment
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate
        n_steps: Steps per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        save_path: Directory to save models
        run_name: Name for this training run
    """
    
    print("=" * 70)
    print("A2C TRAINING - Advantage Actor-Critic")
    print("=" * 70)
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  N Steps: {n_steps}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE Lambda: {gae_lambda}")
    print(f"  Entropy Coef: {ent_coef}")
    print(f"  Value Func Coef: {vf_coef}")
    print(f"  Total Timesteps: {total_timesteps}")
    print(f"  Run Name: {run_name}\n")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    run_path = os.path.join(save_path, run_name)
    os.makedirs(run_path, exist_ok=True)
    
    # Create environment
    env = MuseumGuideEnv()
    env = Monitor(env, os.path.join(run_path, "monitor"))
    
    # Create evaluation environment
    eval_env = MuseumGuideEnv()
    eval_env = Monitor(eval_env, os.path.join(run_path, "eval_monitor"))
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_path,
        log_path=run_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=run_path,
        name_prefix="a2c_checkpoint"
    )
    
    # Create A2C model
    model = A2C(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=os.path.join(run_path, "tensorboard")
    )
    
    print("Starting training...\n")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(run_path, "final_model")
    model.save(final_model_path)
    print(f"\n✓ Model saved to: {final_model_path}")
    
    # Evaluate final model
    print("\nEvaluating final model...")
    evaluate_model(model, eval_env, n_episodes=20)
    
    env.close()
    eval_env.close()
    
    return model, run_path


def evaluate_model(model, env, n_episodes=10):
    """Evaluate trained model"""
    episode_rewards = []
    episode_lengths = []
    artifacts_viewed = []
    final_engagements = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        artifacts_viewed.append(info['artifacts_viewed_count'])
        final_engagements.append(info['engagement_level'])
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Average Artifacts: {np.mean(artifacts_viewed):.1f}/9")
    print(f"  Average Final Engagement: {np.mean(final_engagements):.2f}")
    print(f"  Success Rate (Eng>0.6 & Art>=5): {sum(1 for e, a in zip(final_engagements, artifacts_viewed) if e > 0.6 and a >= 5) / n_episodes * 100:.1f}%")


if __name__ == "__main__":
    # Run multiple configurations
    configs = [
        {"learning_rate": 7e-4, "n_steps": 5, "ent_coef": 0.01, "run_name": "a2c_run1"},
        {"learning_rate": 3e-4, "n_steps": 5, "ent_coef": 0.01, "run_name": "a2c_run2"},
        {"learning_rate": 7e-4, "n_steps": 10, "ent_coef": 0.001, "run_name": "a2c_run3"},
        {"learning_rate": 1e-3, "n_steps": 20, "ent_coef": 0.05, "run_name": "a2c_run4"},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION {i}/{len(configs)}")
        print(f"{'='*70}\n")
        train_a2c(**config)