import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from checkout_analyzer import CheckoutAnalyzer

class CheckoutEnvironment:
    """Simulates the Shopify checkout environment"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the checkout state"""
        self.state = {
            'cart_created': False,
            'items_added': False,
            'customer_info_filled': False,
            'shipping_info_filled': False,
            'payment_info_filled': False,
            'errors': [],
            'current_step': 0
        }
        return self._get_state_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute an action and return new state, reward, done, info"""
        reward = 0
        done = False
        info = {}
        
        try:
            if action == 0 and not self.state['cart_created']:
                self.state['cart_created'] = True
                reward = 20  # Increased reward for successful steps
            elif action == 1 and self.state['cart_created'] and not self.state['customer_info_filled']:
                self.state['customer_info_filled'] = True
                reward = 30
            elif action == 2 and self.state['customer_info_filled'] and not self.state['shipping_info_filled']:
                self.state['shipping_info_filled'] = True
                reward = 30
            elif action == 3 and self.state['shipping_info_filled'] and not self.state['payment_info_filled']:
                self.state['payment_info_filled'] = True
                reward = 50
            else:
                reward = -10  # Smaller penalty for invalid actions
            
            # Check if checkout is complete
            if all([self.state['cart_created'], 
                   self.state['customer_info_filled'],
                   self.state['shipping_info_filled'],
                   self.state['payment_info_filled']]):
                reward += 100  # Bonus for completing checkout
                done = True
                
        except Exception as e:
            self.state['errors'].append(str(e))
            reward = -20
            
        self.state['current_step'] += 1
        if self.state['current_step'] >= 20:  # Max steps
            done = True
            
        return self._get_state_vector(), reward, done, info
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert state dict to vector representation"""
        return np.array([
            int(self.state['cart_created']),
            int(self.state['customer_info_filled']),
            int(self.state['shipping_info_filled']),
            int(self.state['payment_info_filled']),
            len(self.state['errors']),
            self.state['current_step']
        ])

class DQNNetwork(nn.Module):
    """Deep Q-Network for learning checkout process"""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CheckoutAgent:
    def __init__(self):
        self.env = CheckoutEnvironment()
        self.state_size = 6
        self.action_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def train(self, episodes: int = 1000):
        """Train the agent with only final report"""
        training_data = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            # Track episode data
            state_history = [state]
            action_history = []
            reward_history = []
            errors = []
            
            while not done:
                # Choose action
                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(self.action_size)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action = self.model(state_tensor).argmax().item()
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Track data
                state_history.append(next_state)
                action_history.append(action)
                reward_history.append(reward)
                if info.get('error'):
                    errors.append(info['error'])
                
                total_reward += reward
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                
                # Learn from experience
                self._learn()
            
            # Update exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Store episode data
            episode_data = self.collect_episode_data(
                episode=episode,
                state_history=state_history,
                action_history=action_history,
                reward_history=reward_history,
                errors=errors,
                success=(total_reward > 200)
            )
            training_data.append(episode_data)
            
            # Only show basic progress
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {total_reward:.2f}, Steps = {len(state_history)}")
        
        # Generate final comprehensive report
        print("\n=== FINAL CHECKOUT OPTIMIZATION REPORT ===\n")
        analyzer = CheckoutAnalyzer()
        final_report = analyzer.generate_report(training_data)
        print(final_report)
        
        return training_data, final_report
    
    def _learn(self):
        """Learn from stored experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        
        # Convert to numpy arrays first
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Then convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def collect_episode_data(self, episode: int, state_history: List, 
                            action_history: List, reward_history: List,
                            errors: List, success: bool) -> Dict:
        """Collect detailed data about an episode"""
        return {
            'episode': episode,
            'total_reward': sum(reward_history),
            'steps': len(state_history),
            'actions': action_history,
            'errors': errors,
            'success': success,
            'completion_time': len(state_history) * 0.1,  # Assuming 100ms per step
            'state_transitions': state_history
        }

if __name__ == "__main__":
    # Install required packages:
    # pip install torch numpy
    
    agent = CheckoutAgent()
    agent.train(episodes=1000) 