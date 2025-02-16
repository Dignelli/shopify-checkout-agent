import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict

class CheckoutVisualizer:
    def __init__(self):
        # Set style for better-looking graphs
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def create_training_dashboard(self, training_data: List[Dict], save_path: str = "training_analysis.png"):
        """Create a comprehensive dashboard of training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        episodes = [ep['episode'] for ep in training_data]
        rewards = [ep['total_reward'] for ep in training_data]
        steps = [ep['steps'] for ep in training_data]
        success_rate = [1 if ep['success'] else 0 for ep in training_data]

        # 1. Learning Curve (Rewards over time)
        ax1.plot(episodes, rewards, label='Reward', alpha=0.3)
        ax1.plot(self._moving_average(rewards, 20), label='Moving Average')
        ax1.set_title('Learning Curve')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()

        # 2. Steps per Episode
        ax2.plot(episodes, steps, label='Steps', alpha=0.3)
        ax2.plot(self._moving_average(steps, 20), label='Moving Average')
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Number of Steps')
        ax2.legend()

        # 3. Success Rate
        window_size = 50
        success_moving_avg = self._moving_average(success_rate, window_size)
        ax3.plot(episodes[window_size-1:], success_moving_avg)
        ax3.set_title(f'Success Rate (Moving Average, Window={window_size})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim([-0.1, 1.1])

        # 4. Action Distribution
        all_actions = [a for ep in training_data for a in ep['actions']]
        action_counts = pd.Series(all_actions).value_counts()
        ax4.bar(action_counts.index, action_counts.values)
        ax4.set_title('Action Distribution')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average of data"""
        return [sum(data[i:i+window_size])/window_size 
                for i in range(len(data)-window_size+1)] 