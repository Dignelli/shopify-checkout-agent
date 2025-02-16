"""
Tests for the Shopify checkout agent.
"""

import pytest
import numpy as np
from src.agent import ShopifyAgent
from src.environment.test_env import TestCheckoutEnv


def test_agent_training():
    """Test that the agent can learn a simple checkout flow."""
    # Create environment and agent
    env = TestCheckoutEnv()
    agent = ShopifyAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        epsilon_decay=0.95  # Faster decay for testing
    )
    
    # Training loop
    num_episodes = 100
    rewards = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            
            agent.store_transition(
                observation,
                action,
                reward,
                next_observation,
                terminated or truncated
            )
            
            agent.train()
            episode_reward += reward
            observation = next_observation
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
    
    # Check if agent improved
    first_10_avg = np.mean(rewards[:10])
    last_10_avg = np.mean(rewards[-10:])
    
    assert last_10_avg > first_10_avg, \
        f"Agent did not improve: first 10 avg = {first_10_avg}, last 10 avg = {last_10_avg}"


def test_agent_optimal_path():
    """Test that the trained agent can find the optimal checkout path."""
    env = TestCheckoutEnv()
    agent = ShopifyAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        epsilon_start=0.0  # No exploration for testing
    )
    
    # Train agent
    for _ in range(100):
        observation, _ = env.reset()
        while True:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, terminated)
            agent.train()
            if terminated or truncated:
                break
            observation = next_observation
    
    # Test optimal path
    observation, _ = env.reset()
    actions_taken = []
    
    while True:
        action = agent.select_action(observation)
        actions_taken.append(action)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    # Optimal path should be: fill_email -> fill_shipping -> fill_payment -> submit
    assert actions_taken == [0, 1, 2, 3], f"Agent did not find optimal path: {actions_taken}"


def test_agent_error_recovery():
    """Test that the agent can recover from error states."""
    env = TestCheckoutEnv()
    agent = ShopifyAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        epsilon_start=0.1,  # Some exploration for error recovery
        memory_size=1000
    )
    
    # Train agent with error scenarios
    for _ in range(200):
        observation, _ = env.reset()
        while True:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, terminated)
            agent.train()
            if terminated or truncated:
                break
            observation = next_observation
    
    # Test error recovery
    observation, _ = env.reset()
    actions_taken = []
    rewards = []
    
    # Force an error by skipping email
    action = 1  # Try shipping before email
    observation, reward, _, _, _ = env.step(action)
    actions_taken.append(action)
    rewards.append(reward)
    
    # Let agent recover
    while True:
        action = agent.select_action(observation)
        actions_taken.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    
    # Agent should recover and complete checkout
    assert terminated, "Agent failed to complete checkout after error"
    assert sum(rewards) > 0, "Agent failed to achieve positive reward after error"


def test_agent_persistence():
    """Test that the agent can persist and reload its state."""
    import os
    import tempfile
    
    env = TestCheckoutEnv()
    agent = ShopifyAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Train agent
    for _ in range(50):
        observation, _ = env.reset()
        while True:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, terminated)
            agent.train()
            if terminated or truncated:
                break
            observation = next_observation
    
    # Save agent state
    with tempfile.NamedTemporaryFile(delete=False) as f:
        agent.save(f.name)
        
        # Create new agent and load state
        new_agent = ShopifyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        new_agent.load(f.name)
    
    # Test both agents perform similarly
    def evaluate_agent(agent):
        observation, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward
    
    original_reward = evaluate_agent(agent)
    loaded_reward = evaluate_agent(new_agent)
    
    assert abs(original_reward - loaded_reward) < 1.0, \
        f"Loaded agent performs differently: original={original_reward}, loaded={loaded_reward}"
    
    # Cleanup
    os.unlink(f.name)


def test_agent_deterministic_behavior():
    """Test that the agent behaves deterministically when not exploring."""
    env = TestCheckoutEnv()
    agent = ShopifyAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        epsilon_start=0.0  # No exploration
    )
    
    # Train agent
    for _ in range(100):
        observation, _ = env.reset()
        while True:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, terminated)
            agent.train()
            if terminated or truncated:
                break
            observation = next_observation
    
    # Test deterministic behavior
    paths = []
    for _ in range(5):
        observation, _ = env.reset()
        actions = []
        while True:
            action = agent.select_action(observation)
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        paths.append(tuple(actions))
    
    # All paths should be identical
    assert len(set(paths)) == 1, "Agent behavior is not deterministic" 