"""
Test environment for the Shopify checkout agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any


class TestCheckoutEnv(gym.Env):
    """
    A simplified test environment that simulates a Shopify checkout flow.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define the action space
        # 0: fill_email, 1: fill_shipping, 2: fill_payment, 3: submit
        self.action_space = spaces.Discrete(4)
        
        # Define the observation space (simplified checkout state)
        # [email_filled, shipping_filled, payment_filled, error_state]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.state = np.zeros(4, dtype=np.float32)
        self.current_step = 0
        self.error_state = False
        
        return self.state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # Handle different actions
        if action == 0:  # fill_email
            if self.state[0] == 0:  # if email not filled
                self.state[0] = 1
                reward = 1
            else:
                reward = -0.1  # penalty for redundant action
                
        elif action == 1:  # fill_shipping
            if self.state[0] == 1 and self.state[1] == 0:  # if email filled and shipping not filled
                self.state[1] = 1
                reward = 1
            else:
                reward = -0.1
                self.error_state = True
                self.state[3] = 1
                
        elif action == 2:  # fill_payment
            if self.state[1] == 1 and self.state[2] == 0:  # if shipping filled and payment not filled
                self.state[2] = 1
                reward = 1
            else:
                reward = -0.1
                self.error_state = True
                self.state[3] = 1
                
        elif action == 3:  # submit
            if self.state[2] == 1:  # if payment filled
                reward = 5
                terminated = True
            else:
                reward = -1
                self.error_state = True
                self.state[3] = 1
        
        self.current_step += 1
        
        # Truncate if too many steps
        if self.current_step >= 10:
            truncated = True
        
        return self.state, reward, terminated, truncated, info 