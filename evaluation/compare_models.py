"""
Compare All RL Algorithms
Evaluates DQN, REINFORCE, PPO, and A2C on the Museum Guide environment
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import MuseumGuideEnv


class ModelComparator:
    """Compare multiple RL algorithms"""
    
    def __init__(self, n_eval_episodes=100):
        self.n_eval_episodes = n_eval_episodes
        self.results = {}
        
    def load_and_evaluate(self, algorithm: str, model_path: str) -> Dict:
        """
        Load model and evaluate performance
        
        Args:
            algorithm: Algorithm name (dqn, ppo, a2c, reinforce)
            model_path: Path to saved model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {algorithm.upper()}...")
        print(f"Model path: {model_path}")
        
        # Load model based on algorithm
        if algorithm.lower() == 'dqn':
            model = DQN.load(model_path)
        elif algorithm.lower() == 'ppo':
            model = PPO.load(model_path)
        elif algorithm.lower() == 'a2c':
            model = A2C.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create evaluation environment
        env = MuseumGuideEnv(render_mode=None)
        
        # Detailed evaluation
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        engagement_scores = []
        artifacts_viewed = []
        
        for episode in range(self.n_eval_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(steps)
                    engagement_scores.append(info['engagement_level'])
                    artifacts_viewed.append(info['artifacts_viewed_count'])
                    
                    # Success = high engagement and viewed artifacts
                    if info['engagement_level'] > 0.6 and info['artifacts_viewed_count'] >= 4:
                        success_count += 1
        
        # Calculate statistics
        results = {
            'algorithm': algorithm.upper(),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / self.n_eval_episodes,
            'mean_engagement': np.mean(engagement_scores),
            'std_engagement': np.std(engagement_scores),
            'mean_artifacts': np.mean(artifacts_viewed),
            'std_artifacts': np.std(artifacts_viewed),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'engagement_scores': engagement_scores,
            'artifacts_viewed': artifacts_viewed
        }
        
        self.results[algorithm] = results
        
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Mean Engagement: {results['mean_engagement']:.2f}")
        print(f"Mean Artifacts: {results['mean_artifacts']:.1f}")
        
        return results
    
    def compare_all(self, model_paths: Dict[str, str]):
        """
        Compare all algorithms
        
        Args:
            model_paths: Dictionary mapping algorithm names to model paths
        """
        print("="*70)
        print("COMPARING ALL ALGORITHMS")
        print("="*70)
        
        for algorithm, path in model_paths.items():
            if os.path.exists(path):
                self.load_and_evaluate(algorithm, path)
            else:
                print(f"\nWarning: Model not found for {algorithm} at {path}")
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table"""
        data = []
        for algo, results in self.results.items():
            data.append({
                'Algorithm': results['algorithm'],
                'Mean Reward': f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
                'Success Rate (%)': f"{results['success_rate']*100:.1f}",
                'Mean Engagement': f"{results['mean_engagement']:.2f} ± {results['std_engagement']:.2f}",
                'Mean Artifacts': f"{results['mean_artifacts']:.1f} ± {results['std_artifacts']:.1f}",
                'Mean Episode Length': f"{results['mean_length']:.1f}"
            })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_comparisons(self, save_dir='results/graphs'):
        """Generate comparison plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Mean Rewards Comparison
        fig, ax = plt.subplots()
        algorithms = [r['algorithm'] for r in self.results.values()]
        mean_rewards = [r['mean_reward'] for r in self.results.values()]
        std_rewards = [r['std_reward'] for r in self.results.values()]
        
        x = np.arange(len(algorithms))
        ax.bar(x, mean_rewards, yerr=std_rewards, capsize=10, alpha=0.7)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Mean Episode Reward', fontsize=12)
        ax.set_title('Algorithm Performance Comparison - Mean Rewards', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mean_rewards_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Success Rate Comparison
        fig, ax = plt.subplots()
        success_rates = [r['success_rate']*100 for r in self.results.values()]
        bars = ax.bar(x, success_rates, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)])
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Algorithm Performance Comparison - Success Rate', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'success_rate_comparison.png'), dpi=300)
        plt.close()
        
        # 3. Multiple Metrics Radar Chart
        categories = ['Mean Reward\n(Normalized)', 'Success Rate', 'Engagement', 'Artifacts\nViewed']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for algo, results in self.results.items():
            # Normalize values to 0-1 scale
            values = [
                (results['mean_reward'] + 50) / 100,  # Normalize rewards
                results['success_rate'],
                results['mean_engagement'],
                results['mean_artifacts'] / 9  # Normalize to max artifacts
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=results['algorithm'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Multi-Metric Performance Comparison', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Episode Rewards Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (algo, results) in enumerate(self.results.items()):
            if idx < 4:
                axes[idx].hist(results['episode_rewards'], bins=30, alpha=0.7, edgecolor='black')
                axes[idx].axvline(results['mean_reward'], color='red', linestyle='--', 
                                 linewidth=2, label=f"Mean: {results['mean_reward']:.1f}")
                axes[idx].set_xlabel('Episode Reward', fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].set_title(f'{results["algorithm"]} - Reward Distribution', 
                                   fontsize=12, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'reward_distributions.png'), dpi=300)
        plt.close()
        
        # 5. Engagement vs Artifacts Scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, (algo, results) in enumerate(self.results.items()):
            ax.scatter(results['artifacts_viewed'], results['engagement_scores'], 
                      alpha=0.5, s=50, label=results['algorithm'], color=colors[idx])
        
        ax.set_xlabel('Artifacts Viewed', fontsize=12)
        ax.set_ylabel('Final Engagement Score', fontsize=12)
        ax.set_title('Artifacts Viewed vs Visitor Engagement', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'engagement_vs_artifacts.png'), dpi=300)
        plt.close()
        
        print(f"\nPlots saved to: {save_dir}")
    
    def save_results(self, save_path='results/comparison_results.json'):
        """Save comparison results to JSON"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare results for JSON (remove non-serializable numpy arrays)
        results_to_save = {}
        for algo, results in self.results.items():
            results_to_save[algo] = {
                k: v for k, v in results.items() 
                if k not in ['episode_rewards', 'episode_lengths', 'engagement_scores', 'artifacts_viewed']
            }
        
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        print(f"\nResults saved to: {save_path}")


def main():
    # Define model paths
    model_paths = {
        'DQN': 'models/dqn/best_model.zip',
        'REINFORCE': 'models/reinforce/best_model.zip',
        'PPO': 'models/ppo/best_model.zip',
        'A2C': 'models/a2c/best_model.zip'
    }
    
    # Create comparator
    comparator = ModelComparator(n_eval_episodes=100)
    
    # Run comparison
    comparator.compare_all(model_paths)
    
    # Generate and display comparison table
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*70)
    df = comparator.generate_comparison_table()
    print(df.to_string(index=False))
    print("="*70)
    
    # Save table to CSV
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/comparison_table.csv', index=False)
    print("\nComparison table saved to: results/comparison_table.csv")
    
    # Generate plots
    comparator.plot_comparisons()
    
    # Save detailed results
    comparator.save_results()
    
    # Identify best algorithm
    best_algo = max(comparator.results.items(), 
                   key=lambda x: x[1]['mean_reward'])
    print(f"\n{'='*70}")
    print(f"BEST PERFORMING ALGORITHM: {best_algo[0].upper()}")
    print(f"Mean Reward: {best_algo[1]['mean_reward']:.2f}")
    print(f"Success Rate: {best_algo[1]['success_rate']*100:.1f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()