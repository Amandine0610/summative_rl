"""
Main Entry Point - Museum Guide Agent
Runs the best performing model for demonstration
"""

import os
import sys
import argparse
import time
from datetime import datetime

from stable_baselines3 import PPO, DQN, A2C
from environment.custom_env import MuseumGuideEnv


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print("  MUSEUM GUIDE AGENT - REINFORCEMENT LEARNING DEMONSTRATION")
    print("  Ingabo Museum - Visitor Engagement Optimization")
    print("  Author: Amandine Irakoze")
    print("="*80 + "\n")


def print_episode_summary(episode_num, steps, total_reward, engagement, artifacts, time_spent):
    """Print episode summary statistics"""
    print("\n" + "-"*80)
    print(f"EPISODE {episode_num} SUMMARY")
    print("-"*80)
    print(f"  Total Steps:           {steps}")
    print(f"  Total Reward:          {total_reward:.2f}")
    print(f"  Final Engagement:      {engagement:.2f} / 1.00")
    print(f"  Artifacts Viewed:      {artifacts} / 9")
    print(f"  Time Spent:            {time_spent} minutes")
    print(f"  Success:               {'YES ✓' if engagement > 0.6 and artifacts >= 4 else 'NO ✗'}")
    print("-"*80 + "\n")


def run_demonstration(model_path, algorithm='ppo', n_episodes=3, render=True, delay=0.3):
    """
    Run demonstration with trained agent
    
    Args:
        model_path: Path to saved model
        algorithm: Algorithm name (ppo, dqn, a2c)
        n_episodes: Number of episodes to run
        render: Whether to render visualization
        delay: Delay between steps (seconds)
    """
    
    print_banner()
    
    # Load model
    print(f"Loading {algorithm.upper()} model from: {model_path}")
    
    if algorithm.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algorithm.lower() == 'dqn':
        model = DQN.load(model_path)
    elif algorithm.lower() == 'a2c':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"Model loaded successfully!\n")
    
    # Create environment
    render_mode = 'human' if render else None
    env = MuseumGuideEnv(render_mode=render_mode)
    
    # Action descriptions
    action_names = {
        0: "Move North", 1: "Move South", 2: "Move East", 3: "Move West",
        4: "Recommend Cultural", 5: "Recommend Historical", 6: "Recommend Artistic",
        7: "Provide Details", 8: "Switch to Kinyarwanda", 9: "Switch to English",
        10: "Suggest Rest", 11: "End Tour"
    }
    
    # Run episodes
    total_rewards = []
    total_success = 0
    
    for episode in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"STARTING EPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*80}\n")
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Print visitor profile
        print("VISITOR PROFILE:")
        lang_pref = ['English', 'Kinyarwanda', 'Both Languages'][obs['language_pref']]
        print(f"  Language Preference:   {lang_pref}")
        print(f"  Interest - Cultural:   {obs['interest_vector'][0]:.2f}")
        print(f"  Interest - Historical: {obs['interest_vector'][1]:.2f}")
        print(f"  Interest - Artistic:   {obs['interest_vector'][2]:.2f}")
        print(f"  Initial Engagement:    {obs['visitor_engagement'][0]:.2f}")
        print("\n" + "-"*80)
        print("AGENT ACTIONS:")
        print("-"*80 + "\n")
        
        # Episode loop
        while not done:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print step info (verbose mode)
            if steps % 5 == 0 or abs(reward) > 5:  # Print significant steps
                print(f"Step {steps:3d}: {action_names[action]:22s} | "
                      f"Reward: {reward:6.2f} | "
                      f"Engagement: {obs['visitor_engagement'][0]:.2f} | "
                      f"Artifacts: {info['artifacts_viewed_count']}")
            
            # Render
            if render:
                env.render()
                time.sleep(delay)
            
            # Check termination
            if done:
                engagement = info['engagement_level']
                artifacts = info['artifacts_viewed_count']
                time_spent = obs['time_spent'][0]
                
                # Print summary
                print_episode_summary(episode + 1, steps, episode_reward, 
                                    engagement, artifacts, time_spent)
                
                total_rewards.append(episode_reward)
                if engagement > 0.6 and artifacts >= 4:
                    total_success += 1
        
        # Delay between episodes
        if episode < n_episodes - 1 and render:
            print("Press Enter to continue to next episode...")
            input()
    
    # Final statistics
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE - OVERALL STATISTICS")
    print("="*80)
    print(f"  Total Episodes:        {n_episodes}")
    print(f"  Average Reward:        {sum(total_rewards)/n_episodes:.2f}")
    print(f"  Best Reward:           {max(total_rewards):.2f}")
    print(f"  Worst Reward:          {min(total_rewards):.2f}")
    print(f"  Success Rate:          {(total_success/n_episodes)*100:.1f}%")
    print(f"  Algorithm Used:        {algorithm.upper()}")
    print(f"  Demonstration Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Museum Guide Agent - Main Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run PPO model demonstration
  python main.py --model models/ppo/best_model.zip --algorithm ppo
  
  # Run DQN model with 5 episodes
  python main.py --model models/dqn/best_model.zip --algorithm dqn --episodes 5
  
  # Run without visualization (faster)
  python main.py --model models/ppo/best_model.zip --algorithm ppo --no-render
  
  # Run with slower visualization
  python main.py --model models/ppo/best_model.zip --algorithm ppo --delay 0.5
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'dqn', 'a2c', 'reinforce'],
                       help='Algorithm type')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay between steps (seconds)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("\nPlease train a model first using one of the training scripts:")
        print("  python training/ppo_training.py")
        print("  python training/dqn_training.py")
        print("  python training/a2c_training.py")
        return
    
    # Run demonstration
    try:
        run_demonstration(
            model_path=args.model,
            algorithm=args.algorithm,
            n_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay
        )
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        raise


if __name__ == '__main__':
    main()