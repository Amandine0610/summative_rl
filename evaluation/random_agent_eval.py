import csv
import time
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import MuseumGuideEnv


def run_random_agent(env, episodes=200, render=False, delay=0.05, seed=None, save_csv=None):
    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)
    results = []
    for ep in range(1, episodes+1):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        steps = 0
        start = time.time()
        while not done:
            action = env.action_space.sample()
            obs, r, done, _, info = env.step(int(action))
            ep_reward += r
            steps += 1
            if render:
                env.render()
                time.sleep(delay)
        elapsed = time.time() - start
        visitors = info.get('visitors', env.visitors)
        success = all(v['engaged'] for v in visitors)
        results.append({'episode': ep, 'reward': ep_reward, 'steps': steps, 'success': int(success), 'elapsed_s': round(elapsed,3)})
        if ep % max(1, episodes//10) == 0:
            print(f"[{ep}/{episodes}] reward={ep_reward:.3f} steps={steps} success={success}")
    # summary
    import numpy as np
    rewards = np.array([r['reward'] for r in results])
    successes = np.array([r['success'] for r in results])
    steps_arr = np.array([r['steps'] for r in results])
    summary = {'episodes': episodes, 'avg_reward': float(rewards.mean()), 'std_reward': float(rewards.std()), 'avg_steps': float(steps_arr.mean()), 'success_rate': float(successes.mean())}
    print('\n=== RANDOM AGENT SUMMARY ===')
    print(summary)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode','reward','steps','success','elapsed_s'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved to {save_csv}")
    return summary, results

if __name__ == '__main__':
    env = MuseumGuideEnv(grid_size=10, n_visitors=4, max_steps=200)
    run_random_agent(env, episodes=200, render=False, save_csv='results/random_baseline.csv')
    env.close()
    print("\n✓ Random agent evaluation complete!")