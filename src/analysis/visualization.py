"""
Visualization tools for analyzing agent behavior.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import seaborn as sns


def plot_learning_curve(rewards: List[float], window: int = 10) -> None:
    """Plot the learning curve with rolling average."""
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Plot rolling average
    rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(rolling_mean, color='blue', label=f'{window}-Episode Average')
    
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_action_distribution(actions: List[int], num_actions: int) -> None:
    """Plot the distribution of actions taken by the agent."""
    plt.figure(figsize=(8, 6))
    
    action_counts = np.bincount(actions, minlength=num_actions)
    action_names = ['Email', 'Shipping', 'Payment', 'Submit']
    
    sns.barplot(x=action_names, y=action_counts)
    plt.title('Action Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_state_transitions(
    states: List[np.ndarray],
    actions: List[int],
    rewards: List[float]
) -> None:
    """Plot the state transitions and rewards."""
    states_array = np.array(states)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    
    # Plot state components
    for i in range(states_array.shape[1]):
        plt.plot(states_array[:, i], label=f'State {i}')
    
    plt.title('State Transitions')
    plt.legend(['Email', 'Shipping', 'Payment', 'Error'])
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(rewards, color='green')
    plt.title('Rewards')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 