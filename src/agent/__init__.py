"""
This module contains agent-related implementations for the Shopify checkout process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import random
from collections import deque
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DQNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ShopifyAgent:
    """
    A reinforcement learning agent for analyzing Shopify checkout flows.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        device: str = None,
        debug: bool = False
    ):
        """Initialize the Shopify checkout agent."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.action_dim = action_dim
        
        # Metrics tracking
        self.training_steps = 0
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        
        logger.debug("Agent initialized with configuration: %s", {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'memory_size': memory_size,
            'batch_size': batch_size
        })
    
    async def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        if self.debug:
            logger.debug("Stored transition - Action: %d, Reward: %.2f, Done: %s", 
                        action, reward, done)
    
    async def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        self.training_steps += 1
        
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        if self.debug and self.training_steps % 100 == 0:
            logger.debug("Training step %d - Loss: %.4f, Epsilon: %.3f", 
                        self.training_steps, loss_value, self.epsilon)
        
        return loss_value
    
    async def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.debug:
            logger.debug("Target network updated")
    
    def save(self, path: str):
        """Save agent state to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses
        }, path)
        logger.info("Agent saved to %s", path)
    
    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_losses = checkpoint.get('episode_losses', [])
        logger.info("Agent loaded from %s", path)

    async def act(self, state: np.ndarray) -> int:
        """Alias for select_action"""
        return await self.select_action(state)

__all__ = ['ShopifyAgent'] 