import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural Network for Deep Q Learning
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
        self.optimizer = optim.Adam(self.network.parameters())
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, epsilon):
        """Choose action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self._encode_state(state))
            q_values = self.network(state_tensor)
            return torch.argmax(q_values).item()
            
    def train_on_batch(self, batch):
        """Train the network on a batch of experiences"""
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor([self._encode_state(s) for s in states])
        next_states = torch.FloatTensor([self._encode_state(s) for s in next_states])
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q = self.network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _encode_state(self, state):
        """Convert state string to one-hot encoding"""
        encoding = np.zeros(self.state_size)
        state_idx = ['view_product_details', 'select_variant', 'add_to_cart', 
                     'proceed_to_checkout', 'confirm_order'].index(state)
        encoding[state_idx] = 1
        return encoding 