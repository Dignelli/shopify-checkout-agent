"""
Training script for the Shopify checkout agent with visualization.
"""

import os
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

from src.agent import ShopifyAgent
from src.environment.test_env import TestCheckoutEnv
from src.analysis.visualization import (
    plot_learning_curve,
    plot_action_distribution,
    plot_state_transitions
)

def train_agent(
    num_episodes: int = 1000,
    save_dir: str = "checkpoints",
    debug: bool = False
):
    """
    Train the agent and visualize its progress.
    """
    # Create environment and agent
    env = TestCheckoutEnv()
    agent = ShopifyAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        debug=debug
    )
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    all_rewards = []
    all_actions = []
    all_states = []
    episode_lengths = []
    
    # Training loop with progress bar
    progress_bar = tqdm(range(num_episodes), desc="Training")
    for episode in progress_bar:
        states = []
        actions = []
        rewards = []
        
        observation, _ = env.reset()
        states.append(observation)
        
        while True:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            
            # Store transition
            agent.store_transition(
                observation,
                action,
                reward,
                next_observation,
                terminated or truncated
            )
            
            # Store for visualization
            states.append(next_observation)
            actions.append(action)
            rewards.append(reward)
            
            # Train agent
            loss = agent.train()
            
            observation = next_observation
            if terminated or truncated:
                break
        
        # Update metrics
        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)
        all_actions.extend(actions)
        all_states.extend(states)
        episode_lengths.append(len(actions))
        
        # Update progress bar
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'epsilon': f"{agent.epsilon:.3f}",
            'avg_length': f"{np.mean(episode_lengths[-100:]):.1f}"
        })
        
        # Periodically save agent and plot progress
        if (episode + 1) % 100 == 0:
            # Save checkpoint
            agent.save(os.path.join(save_dir, f'checkpoint_{episode+1}.pth'))
            
            # Plot current progress
            plot_learning_curve(all_rewards)
            plot_action_distribution(all_actions, env.action_space.n)
            plot_state_transitions(all_states, all_actions, all_rewards)
    
    # Save final model and metrics
    agent.save(os.path.join(save_dir, 'final_model.pth'))
    
    return agent, {
        'rewards': all_rewards,
        'actions': all_actions,
        'states': all_states,
        'episode_lengths': episode_lengths
    }

def main():
    parser = argparse.ArgumentParser(description='Train Shopify checkout agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    try:
        agent, metrics = train_agent(
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            debug=args.debug
        )
        
        print("\nTraining completed!")
        print(f"Final average reward (last 100 episodes): {np.mean(metrics['rewards'][-100:]):.2f}")
        print(f"Average episode length: {np.mean(metrics['episode_lengths']):.1f} steps")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main() 