from playwright.async_api import async_playwright
from collections import defaultdict
import time
import asyncio

class BrowserCheckoutAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.simulation_results = defaultdict(list)
        self.checkout_paths = [
            {'type': 'guest_checkout', 'payment': 'credit_card'},
            {'type': 'guest_checkout', 'payment': 'shop_pay'},
            {'type': 'account_checkout', 'payment': 'credit_card'},
            {'type': 'mobile_checkout', 'payment': 'shop_pay'}
        ]
    
    async def setup(self):
        """Initialize playwright and browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch()
    
    async def simulate_checkout(self, product_url: str):
        """Run multiple checkout simulations"""
        print("\n🤖 Starting Checkout Simulations...")
        
        if not self.browser:
            await self.setup()
        
        results = {
            'simulations': [],
            'metrics': defaultdict(list),
            'friction_points': []
        }
        
        for path in self.checkout_paths:
            print(f"\nSimulating {path['type']} with {path['payment']}...")
            
            try:
                simulation_result = await self._run_simulation(product_url, path)
                results['simulations'].append(simulation_result)
                
                # Track timing and success rates
                results['metrics']['completion_times'].append(simulation_result['time_taken'])
                results['metrics']['success_rate'].append(simulation_result['success'])
                
                if simulation_result['friction_points']:
                    results['friction_points'].extend(simulation_result['friction_points'])
                
                print(f"✓ Simulation completed: {simulation_result['success']}")
                
            except Exception as e:
                print(f"× Simulation failed: {str(e)}")
                results['friction_points'].append({
                    'step': path['type'],
                    'error': str(e),
                    'severity': 'high'
                })
        
        # Calculate aggregate metrics
        results['summary'] = {
            'avg_completion_time': sum(results['metrics']['completion_times']) / len(results['metrics']['completion_times']),
            'success_rate': sum(results['metrics']['success_rate']) / len(results['metrics']['success_rate']) * 100,
            'total_simulations': len(self.checkout_paths),
            'unique_friction_points': len(set(fp['step'] for fp in results['friction_points']))
        }
        
        return results

    async def _run_simulation(self, product_url: str, path: dict):
        """Run a single checkout simulation"""
        page = await self.browser.new_page()
        start_time = time.time()
        
        if path['type'] == 'mobile_checkout':
            await page.set_viewport_size({"width": 375, "height": 812})
        
        results = {
            'path_type': path['type'],
            'payment_method': path['payment'],
            'success': False,
            'time_taken': 0,
            'friction_points': []
        }
        
        try:
            # Detailed checkout steps with timing
            steps_timing = {}
            
            # Add to cart
            cart_start = time.time()
            await self._add_to_cart(page, product_url)
            steps_timing['add_to_cart'] = time.time() - cart_start
            
            # Initiate checkout
            checkout_start = time.time()
            await self._initiate_checkout(page)
            steps_timing['initiate_checkout'] = time.time() - checkout_start
            
            # Fill customer info
            info_start = time.time()
            await self._fill_customer_info(page, path['type'])
            steps_timing['customer_info'] = time.time() - info_start
            
            # Select payment
            payment_start = time.time()
            await self._handle_payment(page, path['payment'])
            steps_timing['payment'] = time.time() - payment_start
            
            results['success'] = True
            results['steps_timing'] = steps_timing
            
        except Exception as e:
            results['friction_points'].append({
                'step': self._get_current_step(page),
                'error': str(e),
                'timing': time.time() - start_time
            })
        finally:
            results['time_taken'] = time.time() - start_time
            await page.close()
            
        return results

    async def _get_current_step(self, page):
        """Determine current checkout step"""
        try:
            # Check for various page indicators
            if await page.query_selector('#checkout_email'):
                return 'customer_information'
            elif await page.query_selector('#checkout_shipping_address_first_name'):
                return 'shipping_address'
            elif await page.query_selector('#checkout_payment_gateway'):
                return 'payment'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    async def _add_to_cart(self, page, url):
        """Add product to cart"""
        try:
            await page.goto(url)
            # Find first product
            await page.click('.product-card:first-child')
            await page.click('button[name="add"]')
            await page.wait_for_selector('.cart-notification')
        except Exception as e:
            self.friction_points.append({
                'step': 'add_to_cart',
                'error': str(e)
            })
            raise
    
    async def _initiate_checkout(self, page):
        """Start checkout process"""
        try:
            await page.click('a[href="/cart"]')
            await page.wait_for_selector('button[name="checkout"]')
            await page.click('button[name="checkout"]')
        except Exception as e:
            self.friction_points.append({
                'step': 'initiate_checkout',
                'error': str(e)
            })
            raise
    
    async def _fill_customer_info(self, page, path_type):
        """Fill customer information"""
        try:
            # Wait for customer information form
            await page.wait_for_selector('#checkout_email')
            
            # Fill form
            await page.fill('#checkout_email', 'test@example.com')
            await page.fill('#checkout_shipping_address_first_name', 'Test')
            await page.fill('#checkout_shipping_address_last_name', 'User')
            await page.fill('#checkout_shipping_address_address1', '123 Test St')
            await page.fill('#checkout_shipping_address_city', 'New York')
            await page.fill('#checkout_shipping_address_zip', '10001')
            await page.fill('#checkout_shipping_address_phone', '1234567890')
            
            # Continue to shipping
            await page.click('#continue_button')
        except Exception as e:
            self.friction_points.append({
                'step': 'customer_info',
                'error': str(e)
            })
            raise
    
    async def _handle_payment(self, page, payment_method):
        """Handle payment process"""
        try:
            # Wait for shipping method
            await page.wait_for_selector('#continue_button')
            await page.click('#continue_button')
            
            # Wait for payment method
            await page.wait_for_selector('#payment-method')
            # Note: We don't actually submit payment in test mode
        except Exception as e:
            self.friction_points.append({
                'step': 'payment',
                'error': str(e)
            })
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop() 