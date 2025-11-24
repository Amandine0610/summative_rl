"""
Quick Visual Demo - See Your Agent in Action
This shows the environment with a random agent (no training needed)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import MuseumGuideEnv

def quick_demo():
    """Run a quick visual demo"""
    print("=" * 70)
    print("QUICK VISUAL DEMO - Random Agent")
    print("=" * 70)
    print("\nA window will open showing the museum grid.")
    print("The RED circle is your agent.")
    print("Colored circles are exhibits.")
    print("Press Ctrl+C to stop.\n")
    
    # Create environment with visualization
    env = MuseumGuideEnv(render_mode='human')
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Visitor Profile:")
    print(f"  Language: {['English', 'Kinyarwanda', 'Both'][obs['language_pref']]}")
    print(f"  Interests: Cultural={obs['interest_vector'][0]:.2f}, "
          f"Historical={obs['interest_vector'][1]:.2f}, "
          f"Artistic={obs['interest_vector'][2]:.2f}")
    print(f"\nAgent is taking random actions...\n")
    
    episode_reward = 0
    steps = 0
    episode_count = 0
    
    try:
        while True:  # Run indefinitely until user stops
            # Take random action (exclude action 11 for demo purposes)
            action = env.action_space.sample()
            
            # Skip "end tour" action to see full exploration
            # (In real training, agent will learn when to use it)
            while action == 11:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Render the environment
            env.render()
            
            # Print updates every 10 steps
            if steps % 10 == 0:
                print(f"Step {steps}: Engagement={obs['visitor_engagement'][0]:.2f}, "
                      f"Artifacts={info['artifacts_viewed_count']}, Reward={episode_reward:.1f}")
            
            # Slow down for visibility
            time.sleep(0.2)
            
            if terminated or truncated:
                episode_count += 1
                print(f"\n{'='*50}")
                print(f"Episode {episode_count} ended after {steps} steps")
                print(f"Final Engagement: {info['engagement_level']:.2f}")
                print(f"Artifacts Viewed: {info['artifacts_viewed_count']}/9")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"{'='*50}")
                
                # Ask to continue
                response = input("\nRun another episode? (y/n): ")
                if response.lower() != 'y':
                    break
                
                print("\nStarting new episode...")
                obs, info = env.reset()
                
                print(f"Visitor Profile:")
                print(f"  Language: {['English', 'Kinyarwanda', 'Both'][obs['language_pref']]}")
                print(f"  Interests: Cultural={obs['interest_vector'][0]:.2f}, "
                      f"Historical={obs['interest_vector'][1]:.2f}, "
                      f"Artistic={obs['interest_vector'][2]:.2f}\n")
                
                episode_reward = 0
                steps = 0
    
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    
    finally:
        env.close()
        print("\n✓ Demo complete!")

if __name__ == "__main__":
    quick_demo()