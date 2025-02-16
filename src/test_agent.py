"""
Test script for the Shopify agent without browser interaction
"""
import asyncio
import numpy as np
from agent import ShopifyAgent

async def test_agent():
    # Create a simple test environment state
    state_size = 4
    action_size = 5
    test_state = np.random.rand(state_size).astype(np.float32)  # Convert to float32
    
    # Initialize agent with test config
    config = {
        'agent': {
            'learning_rate': 1e-4
        }
    }
    
    agent = ShopifyAgent(state_size, action_size, config)
    
    # Test action selection
    action = await agent.act(test_state)
    print(f"Selected action: {action}")
    
    # Test storing transitions
    next_state = np.random.rand(state_size).astype(np.float32)  # Convert to float32
    agent.store_transition(test_state, action, 1.0, next_state, False)
    
    # Test training step
    for _ in range(32):  # Fill memory with some transitions
        state = np.random.rand(state_size).astype(np.float32)  # Convert to float32
        next_state = np.random.rand(state_size).astype(np.float32)  # Convert to float32
        action = np.random.randint(action_size)
        agent.store_transition(state, action, 1.0, next_state, False)
    
    loss = await agent.train_step()
    print(f"Training loss: {loss}")

if __name__ == "__main__":
    asyncio.run(test_agent()) 