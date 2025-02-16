"""
Environment for interacting with Shopify checkout process.
"""

import os
import logging
from typing import Dict, Any, Tuple, List
from pyppeteer import launch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyppeteer.chromium_downloader import chromium_executable
import asyncio
from abc import ABC, abstractmethod
import shopify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseShopifyEnvironment(ABC):
    @abstractmethod
    async def initialize(self):
        """Initialize the environment"""
        pass
    
    @abstractmethod
    async def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    async def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (state, reward, terminated, truncated, info)"""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass

class BrowserShopifyEnvironment(BaseShopifyEnvironment):
    def __init__(self, config: Dict[str, Any]):
        self.store_url = os.getenv('SHOPIFY_STORE_URL')
        self.browser = None
        self.page = None
        self.config = config
        self.current_step = 0
        self.max_steps = 50
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the browser and page"""
        try:
            logger.info("Checking Chromium installation...")
            chromium_path = chromium_executable()
            if not os.path.exists(chromium_path):
                logger.info("Chromium not found, downloading...")
                from pyppeteer.chromium_downloader import download_chromium
                await download_chromium()
                logger.info("Chromium download completed")
            
            logger.info("Launching browser...")
            launch_args = {
                'headless': self.config['environment']['headless'],
                'args': [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu',
                    '--no-zygote',
                    '--single-process'
                ],
                'handleSIGINT': False,
                'handleSIGTERM': False,
                'handleSIGHUP': False
            }
            
            logger.info(f"Launch arguments: {launch_args}")
            
            self.browser = await launch(**launch_args)
            logger.info("Browser launched successfully")
            
            logger.info("Creating new page...")
            self.page = await self.browser.newPage()
            await self.page.setViewport({'width': 1280, 'height': 800})
            logger.info("Page created successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            if self.browser:
                await self.browser.close()
            raise

    async def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        self.current_step = 0
        try:
            await self.page.goto(self.store_url)
            await self.clear_cart()
            state = await self._get_state()
            return state, {}
        except Exception as e:
            logger.error(f"Reset failed: {str(e)}")
            raise

    async def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            Tuple containing:
            - observation (np.ndarray)
            - reward (float)
            - terminated (bool)
            - truncated (bool)
            - info (Dict)
        """
        self.current_step += 1
        try:
            # Execute action
            reward = await self._execute_action(action)
            
            # Get new state
            next_state = await self._get_state()
            
            # Check if episode is done
            terminated = await self._is_checkout_complete() or self.current_step >= self.max_steps
            truncated = False
            
            # Additional info
            info = {
                'action_taken': action,
                'page_url': self.page.url,
                'step': self.current_step
            }
            
            return next_state, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            return await self._get_state(), -1, True, False, {"error": str(e)}

    async def _get_state(self) -> np.ndarray:
        """Get the current state observation."""
        try:
            # Get current URL to determine stage
            current_url = self.page.url
            
            # Create state vector [cart, checkout, thank_you, step_progress]
            state = np.zeros(4, dtype=np.float32)
            
            # Set flags based on current page
            state[0] = 1.0 if "cart" in current_url else 0.0
            state[1] = 1.0 if "checkout" in current_url else 0.0
            state[2] = 1.0 if "thank_you" in current_url else 0.0
            state[3] = float(self.current_step) / self.max_steps  # Progress through episode
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting state: {str(e)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    async def _execute_action(self, action: int) -> float:
        """
        Execute the given action and return the reward.
        
        Args:
            action (int): Action to take
            
        Returns:
            float: Reward for the action
        """
        try:
            if action == 0:  # add_to_cart
                return await self.click_add_to_cart()
            elif action == 1:  # checkout
                return await self.proceed_to_checkout()
            elif action == 2:  # shipping
                return await self.fill_shipping_info()
            elif action == 3:  # payment
                return await self.fill_payment_info()
            elif action == 4:  # confirm
                return await self.confirm_order()
            
            return -0.1  # Small penalty for invalid actions
            
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            return -1.0

    async def _is_checkout_complete(self) -> bool:
        """Check if checkout is complete."""
        try:
            # Look for success indicators
            success_elements = await self.page.querySelectorAll('.order-confirmation, .thank-you-page')
            return len(success_elements) > 0
        except Exception as e:
            logger.error(f"Error checking checkout completion: {str(e)}")
            return False

    async def close(self):
        """Clean up resources."""
        if self.browser:
            try:
                logger.info("Closing browser...")
                await self.browser.close()
                logger.info("Browser closed successfully")
                self.browser = None
                self.page = None
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                raise

    async def click_add_to_cart(self) -> float:
        """Click the Add to Cart button"""
        try:
            # Wait for and find Add to Cart button
            add_to_cart_button = await self.page.waitForSelector(
                'button[name="add"], button.add-to-cart, input[name="add"]',
                {'visible': True, 'timeout': 5000}
            )
            await add_to_cart_button.click()
            
            # Verify item was added to cart
            await self.page.waitForSelector('.cart-count, .cart-item', 
                                         {'visible': True, 'timeout': 5000})
            return 1.0
        except Exception as e:
            logger.error(f"Failed to add to cart: {str(e)}")
            return -1.0

    async def proceed_to_checkout(self) -> float:
        """Navigate to checkout page"""
        try:
            # Find and click checkout button
            checkout_button = await self.page.waitForSelector(
                'a[href*="checkout"], button[name="checkout"]',
                {'visible': True, 'timeout': 5000}
            )
            await checkout_button.click()
            
            # Verify we reached checkout page
            await self.page.waitForSelector('#checkout, .checkout', 
                                         {'visible': True, 'timeout': 5000})
            return 1.0
        except Exception as e:
            logger.error(f"Failed to proceed to checkout: {str(e)}")
            return -1.0

    async def fill_shipping_info(self) -> float:
        """Fill shipping information form"""
        try:
            # Fill shipping form fields
            form_data = {
                'email': 'test@example.com',
                'firstName': 'Test',
                'lastName': 'User',
                'address1': '123 Test St',
                'city': 'Test City',
                'country': 'United States',
                'state': 'New York',
                'zip': '10001',
                'phone': '1234567890'
            }
            
            for field, value in form_data.items():
                input_selector = f'input[name*="{field}"], input[id*="{field}"]'
                try:
                    input_field = await self.page.waitForSelector(input_selector, 
                                                               {'visible': True, 'timeout': 2000})
                    await input_field.type(value)
                except:
                    logger.warning(f"Could not find field: {field}")
            
            # Click continue button
            continue_button = await self.page.waitForSelector(
                'button[type="submit"], button.continue-button',
                {'visible': True, 'timeout': 5000}
            )
            await continue_button.click()
            return 1.0
        except Exception as e:
            logger.error(f"Failed to fill shipping info: {str(e)}")
            return -1.0

    async def fill_payment_info(self) -> float:
        """Fill payment information form"""
        try:
            # Wait for payment iframe and switch to it
            payment_frame = await self.page.waitForSelector(
                'iframe[name*="card"], iframe[id*="card"]',
                {'visible': True, 'timeout': 5000}
            )
            frame = await payment_frame.contentFrame()
            
            # Fill card details
            await frame.type('input[name*="number"]', '4242424242424242')
            await frame.type('input[name*="exp"]', '1225')
            await frame.type('input[name*="cvc"]', '123')
            
            # Switch back to main frame
            await self.page.evaluate('document.activeElement.blur()')
            return 1.0
        except Exception as e:
            logger.error(f"Failed to fill payment info: {str(e)}")
            return -1.0

    async def confirm_order(self) -> float:
        """Submit the order"""
        try:
            # Find and click the final submit button
            submit_button = await self.page.waitForSelector(
                'button[type="submit"], button.submit-button',
                {'visible': True, 'timeout': 5000}
            )
            await submit_button.click()
            
            # Wait for confirmation page
            await self.page.waitForSelector(
                '.order-confirmation, .thank-you-page',
                {'visible': True, 'timeout': 10000}
            )
            return 5.0  # Higher reward for successful completion
        except Exception as e:
            logger.error(f"Failed to confirm order: {str(e)}")
            return -1.0

    async def clear_cart(self):
        """Clear shopping cart"""
        try:
            await self.page.goto(f"{self.store_url}/cart/clear")
            logger.info("Cart cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cart: {str(e)}")
            raise

class APIShopifyEnvironment(BaseShopifyEnvironment):
    def __init__(self, config: Dict[str, Any]):
        self.store_url = os.getenv('SHOPIFY_STORE_URL')
        self.api_key = os.getenv('SHOPIFY_API_KEY')
        self.api_secret = os.getenv('SHOPIFY_API_SECRET')
        self.config = config
        self.current_step = 0
        self.max_steps = 50
        self.session = None
        self.cart = None
        self.checkout = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize Shopify API session"""
        try:
            shop_url = self.store_url.replace('https://', '')
            self.session = shopify.Session(shop_url, '2023-01', self.api_secret)
            shopify.ShopifyResource.activate_session(self.session)
            self.logger.info("API session initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize API session: {str(e)}")
            raise

    async def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        self.current_step = 0
        try:
            # Clear existing cart/checkout
            self.cart = None
            self.checkout = None
            return await self._get_state(), {}
        except Exception as e:
            self.logger.error(f"Reset failed: {str(e)}")
            raise

    async def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action using API"""
        self.current_step += 1
        try:
            # Execute action
            reward = await self._execute_action(action)
            
            # Get new state
            next_state = await self._get_state()
            
            # Check if episode is done
            terminated = await self._is_checkout_complete() or self.current_step >= self.max_steps
            truncated = False
            
            # Additional info
            info = {
                'action_taken': action,
                'step': self.current_step,
                'cart_token': self.cart.token if self.cart else None,
                'checkout_token': self.checkout.token if self.checkout else None
            }
            
            return next_state, reward, terminated, truncated, info
            
        except Exception as e:
            self.logger.error(f"Step failed: {str(e)}")
            return await self._get_state(), -1, True, False, {"error": str(e)}

    async def _get_state(self) -> np.ndarray:
        """Get current state from API"""
        try:
            features = [
                1.0 if self.cart else 0.0,
                1.0 if self.checkout else 0.0,
                1.0 if self.checkout and self.checkout.completed_at else 0.0,
                float(self.current_step) / self.max_steps
            ]
            return np.array(features, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Failed to get state: {str(e)}")
            return np.zeros(4, dtype=np.float32)

    async def close(self):
        """Clean up API session"""
        if self.session:
            shopify.ShopifyResource.clear_session()
            self.session = None
            self.logger.info("API session closed")

    async def _execute_action(self, action: int) -> float:
        """Execute action using Shopify API"""
        try:
            if action == 0:  # add_to_cart
                return await self._add_to_cart()
            elif action == 1:  # create_checkout
                return await self._create_checkout()
            elif action == 2:  # update_shipping
                return await self._update_shipping()
            elif action == 3:  # update_payment
                return await self._update_payment()
            elif action == 4:  # complete_checkout
                return await self._complete_checkout()
            return -0.1  # Invalid action
        except Exception as e:
            self.logger.error(f"Action execution failed: {str(e)}")
            return -1.0

    async def _is_checkout_complete(self) -> bool:
        """Check if checkout is complete"""
        return bool(self.checkout and self.checkout.completed_at)

    # API-specific action methods
    async def _add_to_cart(self) -> float:
        """Add item to cart using API"""
        try:
            self.cart = shopify.Cart.create({
                'items': [{
                    'variant_id': 'sample_variant_id',  # Replace with actual variant ID
                    'quantity': 1
                }]
            })
            return 1.0
        except Exception as e:
            self.logger.error(f"Failed to add to cart: {str(e)}")
            return -1.0

    async def _create_checkout(self) -> float:
        """Create checkout from cart using API"""
        try:
            if not self.cart:
                return -1.0
            
            self.checkout = shopify.Checkout.create({
                'cart_token': self.cart.token
            })
            return 1.0
        except Exception as e:
            self.logger.error(f"Failed to create checkout: {str(e)}")
            return -1.0

    async def _update_shipping(self) -> float:
        """Update shipping information using API"""
        try:
            if not self.checkout:
                return -1.0
            
            self.checkout.shipping_address = {
                'first_name': 'Test',
                'last_name': 'User',
                'address1': '123 Test St',
                'city': 'Test City',
                'province': 'NY',
                'country': 'US',
                'zip': '10001',
                'phone': '1234567890'
            }
            self.checkout.save()
            return 1.0
        except Exception as e:
            self.logger.error(f"Failed to update shipping: {str(e)}")
            return -1.0

    async def _update_payment(self) -> float:
        """Update payment information using API"""
        try:
            if not self.checkout:
                return -1.0
            
            # Note: In practice, you'd use a payment gateway
            self.checkout.payment = {
                'credit_card': {
                    'number': '4242424242424242',
                    'month': '12',
                    'year': '25',
                    'verification_value': '123'
                }
            }
            self.checkout.save()
            return 1.0
        except Exception as e:
            self.logger.error(f"Failed to update payment: {str(e)}")
            return -1.0

    async def _complete_checkout(self) -> float:
        """Complete checkout using API"""
        try:
            if not self.checkout:
                return -1.0
            
            self.checkout.complete()
            return 5.0 if self.checkout.completed_at else -1.0
        except Exception as e:
            self.logger.error(f"Failed to complete checkout: {str(e)}")
            return -1.0 