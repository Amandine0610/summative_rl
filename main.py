"""
Main entry point - Run best performing model
"""

import os
import sys
import numpy as np
import pygame  # For events
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import MuseumGuideEnv
import time

def run_best_model(model_path, model_type="dqn", n_episodes=3, epsilon=0.1):
    """
    Run the best performing model with optional epsilon for exploration.
    """
    
    print("=" * 70)
    print("RUNNING BEST PERFORMING MODEL (POST-PATCH)")
    print("=" * 70)
    print(f"\nModel Type: {model_type.upper()}")
    print(f"Model Path: {model_path}")
    print(f"Episodes: {n_episodes}, Epsilon: {epsilon}\n")
    
    env = MuseumGuideEnv(render_mode='human')
    
    model = None
    if model_path:
        if model_type.lower() == 'dqn':
            model = DQN.load(model_path)
        elif model_type.lower() == 'ppo':
            model = PPO.load(model_path)
        elif model_type.lower() == 'a2c':
            model = A2C.load(model_path)
        print("✓ Model loaded!\n")
    else:
        print("Using random policy.\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        lang_idx = int(obs['language_pref'])
        print(f"\n{'='*50}")
        print(f"EPISODE {episode + 1}/{n_episodes} | Pref: {['EN', 'RW', 'Both'][lang_idx]}")
        print(f"{'='*50}")
        print(f"Interests: C={obs['interest_vector'][0]:.2f} H={obs['interest_vector'][1]:.2f} A={obs['interest_vector'][2]:.2f}")
        
        while not done:
            # Action: Model or random
            if model:
                action, _ = model.predict(obs, deterministic=True)
                if np.random.rand() < epsilon:  # NEW: 10% random for diversity
                    action = env.action_space.sample()
                    print(f"  [EPSILON] Switched to random action {action}")
            else:
                action = env.action_space.sample()
            
            # Step
            prev_pos = obs['agent_pos'].copy()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Breakdown print (every 10 or big reward)
            at_ex = env._check_exhibit_proximity()
            if steps % 10 == 0 or abs(reward) > 1 or at_ex is not None:
                print(f"Step {steps}: Act={action} | Pos={obs['agent_pos']} (moved? {not np.array_equal(prev_pos, obs['agent_pos'])}) | Eng={obs['visitor_engagement'][0]:.3f} | R={reward:.1f} (cum={episode_reward:.1f}) | Ex={at_ex} | Art={info['artifacts_viewed_count']}")
            
            env.render()
            # Events to avoid freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            time.sleep(0.2)
            
            if done and steps < 50:
                print(f"Early done at step {steps} (low eng? {obs['visitor_engagement'][0]:.2f})")
        
        print(f"\nCOMPLETE: Steps={steps} | Reward={episode_reward:.1f} | Eng={info['engagement_level']:.2f} | Art={info['artifacts_viewed_count']}/9 | ✓ {'High eng +5 art' if info['engagement_level'] > 0.6 and info['artifacts_viewed_count'] >= 5 else 'Needs work'}")
        if episode < n_episodes - 1:
            input("\nEnter for next...")
    
    env.close()
    print("\nDemo done!")

if __name__ == "__main__":
    # Post-patch model (or "" for random)
    run_best_model(
        model_path="models/dqn/dqn_run1/best_model",  # Your patched retrain
        model_type="dqn",
        n_episodes=10,
        epsilon=0.2  # Higher for demo variety
    )