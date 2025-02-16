"""
This module contains analysis tools and utilities for processing simulation results.
"""

import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from src.agent import ShopifyAgent
from src.environment import ShopifyCheckoutEnv


class CheckoutAnalyzer:
    """
    Analyzes and trains the Shopify checkout agent.
    """
    
    def __init__(self, shop_url: str):
        """
        Initialize the analyzer with a shop URL.
        
        Args:
            shop_url (str): URL of the Shopify store to analyze
        """
        self.env = ShopifyCheckoutEnv(shop_url)
        self.agent = ShopifyAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n
        )
        self.rewards_history: List[float] = []
        
    def train(self, num_episodes: int = 100) -> List[float]:
        """
        Train the agent on the checkout environment.
        
        Args:
            num_episodes (int): Number of training episodes
            
        Returns:
            List[float]: History of episode rewards
        """
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                observation = next_observation
                
                if terminated or truncated:
                    break
            
            self.rewards_history.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
            
        return self.rewards_history
    
    def plot_training_progress(self):
        """Plot the training rewards over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()
    
    def analyze_checkout_flow(self) -> Tuple[float, List[str]]:
        """
        Analyze the checkout flow using the trained agent.
        
        Returns:
            Tuple[float, List[str]]: Average reward and list of identified issues
        """
        observation, _ = self.env.reset()
        total_reward = 0
        issues = []
        
        while True:
            action = self.agent.select_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            
            if reward < 0:
                issues.append(f"Friction point detected: {info.get('error', 'Unknown error')}")
                
            if terminated or truncated:
                break
        
        return total_reward, issues 