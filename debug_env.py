import numpy as np
from environment.custom_env import MuseumGuideEnv

env = MuseumGuideEnv()  # No render for speed
obs, info = env.reset()
total_reward = 0
steps = 0
done = False
print("DEBUG SIM - Random Episode")
print(f"Start Pos: {obs['agent_pos']}, Engagement: {obs['visitor_engagement'][0]:.2f}")

while not done:
    action = env.action_space.sample()  # Random
    print(f"Step {steps}: Action={action}, Pos={obs['agent_pos']}, Engagement={obs['visitor_engagement'][0]:.2f}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    done = terminated or truncated
    
    # Print if at exhibit or reward spike
    at_ex = env._check_exhibit_proximity()
    if at_ex is not None:
        print(f"  -> At exhibit {at_ex}! Viewed? {env.artifacts_viewed[at_ex]} | +2 bonus? {reward > 0}")
    if abs(reward) > 1:
        print(f"  -> Reward detail: {reward}")
    
    if steps >= 5:  # Limit for test
        break  # Short run

print(f"\nShort Episode End: Steps={steps}, Reward={total_reward:.2f}, Artifacts={info['artifacts_viewed_count']}, Engagement={info['engagement_level']:.2f}")
env.close()