"""
Main entry point - Run best performing model
"""

import os
import sys
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import MuseumGuideEnv
import time

def run_best_model(model_path, model_type="ppo", n_episodes=5):
    """
    Run the best performing model
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('dqn', 'ppo', 'a2c')
        n_episodes: Number of episodes to run
    """
    
    print("=" * 70)
    print("RUNNING BEST PERFORMING MODEL")
    print("=" * 70)
    print(f"\nModel Type: {model_type.upper()}")
    print(f"Model Path: {model_path}")
    print(f"Episodes to Run: {n_episodes}\n")
    
    # Create environment with rendering
    env = MuseumGuideEnv(render_mode='human')
    
    # Load model
    if model_type.lower() == 'dqn':
        model = DQN.load(model_path)
    elif model_type.lower() == 'ppo':
        model = PPO.load(model_path)
    elif model_type.lower() == 'a2c':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("✓ Model loaded successfully!\n")
    print("Starting demonstration...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\n{'='*50}")
        print(f"EPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*50}")
        print(f"Visitor Profile:")
        print(f"  Language: {['English', 'Kinyarwanda', 'Both'][obs['language_pref']]}")
        print(f"  Interests: Cultural={obs['interest_vector'][0]:.2f}, "
              f"Historical={obs['interest_vector'][1]:.2f}, "
              f"Artistic={obs['interest_vector'][2]:.2f}\n")
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Render
            env.render()
            
            # Print progress
            if steps % 20 == 0:
                print(f"Step {steps}: Engagement={obs['visitor_engagement'][0]:.2f}, "
                      f"Artifacts={info['artifacts_viewed_count']}, "
                      f"Reward={episode_reward:.1f}")
            
            # Slow down for visibility
            time.sleep(0.1)
        
        print(f"\n{'='*50}")
        print(f"EPISODE {episode + 1} COMPLETE")
        print(f"{'='*50}")
        print(f"  Total Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Engagement: {info['engagement_level']:.2f}")
        print(f"  Artifacts Viewed: {info['artifacts_viewed_count']}/9")
        print(f"  Success: {'✓' if info['engagement_level'] > 0.6 and info['artifacts_viewed_count'] >= 5 else '✗'}")
        
        if episode < n_episodes - 1:
            input("\nPress Enter to continue to next episode...")
    
    env.close()
    print("\n✓ Demonstration complete!")


if __name__ == "__main__":
    # Example: Run your best model
    # Update these paths to your actual best model
    
    # Option 1: PPO (usually best for this type of problem)
    run_best_model(
        model_path="models/ppo/ppo_run1/best_model",
        model_type="ppo",
        n_episodes=3
    )
    
    # Option 2: DQN
    # run_best_model(
    #     model_path="models/dqn/dqn_run1/best_model",
    #     model_type="dqn",
    #     n_episodes=3
    # )
    
    # Option 3: A2C
    # run_best_model(
    #     model_path="models/a2c/a2c_run1/best_model",
    #     model_type="a2c",
    #     n_episodes=3
    # )