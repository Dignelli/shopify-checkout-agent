"""
Main script for running the Shopify checkout agent
"""
import os
import json
import asyncio
from dotenv import load_dotenv
import logging
from environment.shopify_env import ShopifyEnvironment
from agent import ShopifyAgent

async def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load environment variables and config
    load_dotenv()
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize environment
    env = ShopifyEnvironment(config)
    await env.initialize()
    
    try:
        # Initialize agent
        state_size = 4  # [cart, checkout, thank_you, step_progress]
        action_size = 5  # [add_to_cart, checkout, shipping, payment, confirm]
        agent = ShopifyAgent(state_size, action_size, config)
        
        # Training loop
        n_episodes = 10
        batch_size = 32
        
        for episode in range(n_episodes):
            state, _ = await env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Agent selects action
                action = await agent.act(state)
                
                # Execute action in environment
                next_state, reward, done, truncated, info = await env.step(action)
                
                # Store experience
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train on past experiences
                if len(agent.memory) > batch_size:
                    loss = await agent.train_step()
                    logger.info(f"Episode {episode + 1}, Loss: {loss:.4f}")
                
                total_reward += reward
                state = next_state
                
                if done or truncated:
                    break
            
            logger.info(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")
            
            # Update target network every 5 episodes
            if episode % 5 == 0:
                await agent.update_target_network()
    
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main()) 